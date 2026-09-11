import pickle
import numpy as np
from scipy.ndimage import maximum_filter
 
 
class AudioMatcher:
    """
    Spectrogram-fingerprint-based audio matcher.
 
    Holds a reference library (built via add_reference) and matches new
    query/mixture spectrograms against it (via detect). fingerprint_kwargs
    passed to __init__ are used consistently for every reference AND every
    query -- required, since hashes from different fan_out/time_delta
    settings are not comparable to each other.
    """
 
    def __init__(self, **fingerprint_kwargs):
        self.fingerprint_kwargs = fingerprint_kwargs
        self.reference_index = {}       
        self.reference_hash_counts = {} 

 
    def extract_keypoints(
        self,
        spectrogram: np.ndarray,
        neighborhood_size=(15, 15),
        amplitude_floor_db=-40.0,
        num_zones_freq=4,
        num_zones_time=8,
        peaks_per_zone=5,
    ):
        """
        Extract robust local-maximum keypoints from a log-magnitude spectrogram.
 
        Parameters
        ----------
        spectrogram : np.ndarray, shape (n_freq, n_time)
            Log-magnitude (dB) spectrogram. Must already be dB-scaled -- peak
            picking on raw linear magnitude over-weights a few very loud bins
            and misses everything else, since audio energy is roughly
            log-distributed.
        neighborhood_size : (freq_span, time_span)
            Size of the local window used to decide "is this point the
            loudest nearby?". Larger -> fewer, more separated peaks. Tune
            against the shortest/most tightly-spaced feature you need to
            resolve (e.g. a fast trill needs a smaller time_span than a
            sustained tone).
        amplitude_floor_db : float
            Peaks quieter than (global_max_db + amplitude_floor_db) are
            discarded. Keeps you from fingerprinting the noise floor in
            silent or near-silent regions.
        num_zones_freq, num_zones_time : int
            The spectrogram is divided into a grid of this many zones. Peaks
            are capped *per zone*, not globally -- without this, nearly all
            peaks come from the single loudest moment/frequency band, and
            quieter but still real parts of the call get no representation
            at all (the original Shazam paper's justification for zoning).
        peaks_per_zone : int
            Max number of peaks kept per zone, by magnitude.
 
        Returns
        -------
        list of (time_idx, freq_idx, magnitude) tuples, unsorted.
        """
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D (freq, time)")
 
        n_freq, n_time = spectrogram.shape
 
        # A point is a local max if it equals the max of its neighborhood.
        # mode='constant', cval=-inf ensures edge bins aren't falsely
        # treated as maxima just because the filter has nothing real to
        # compare them to.
        local_max = maximum_filter(
            spectrogram, size=neighborhood_size, mode="constant", cval=-np.inf
        )
        is_peak = spectrogram == local_max
 
        global_max = spectrogram.max()
        is_loud_enough = spectrogram >= (global_max + amplitude_floor_db)
 
        candidate_mask = is_peak & is_loud_enough
        freq_idxs, time_idxs = np.nonzero(candidate_mask)
        magnitudes = spectrogram[freq_idxs, time_idxs]
 
        if len(magnitudes) == 0:
            return []
 
        freq_zone_edges = np.linspace(0, n_freq, num_zones_freq + 1)
        time_zone_edges = np.linspace(0, n_time, num_zones_time + 1)
        freq_zone_ids = np.digitize(freq_idxs, freq_zone_edges[1:-1])
        time_zone_ids = np.digitize(time_idxs, time_zone_edges[1:-1])
        zone_ids = freq_zone_ids * num_zones_time + time_zone_ids
 
        keypoints = []
        for zone in np.unique(zone_ids):
            in_zone = np.nonzero(zone_ids == zone)[0]
            if len(in_zone) > peaks_per_zone:
                top = in_zone[np.argsort(magnitudes[in_zone])[-peaks_per_zone:]]
            else:
                top = in_zone
            for i in top:
                keypoints.append(
                    (int(time_idxs[i]), int(freq_idxs[i]), float(magnitudes[i]))
                )
 
        return keypoints
 
    @staticmethod
    def _combine_hash(freq1: int, freq2: int, delta_t: int, freq_bits=10, time_bits=10) -> int:
        """
        Pack (freq1, freq2, delta_t) into a single integer hash.
 
        A @staticmethod because it's a pure function of its arguments --
        no dependence on matcher state. Bit-packing (rather than a
        general-purpose hash like hashlib) is deliberate: collision-free by
        construction within the bit budget, faster, and usable directly as
        a dict key with no extra hashing overhead.
 
        freq_bits / time_bits cap how many distinct frequency bins /
        time-delta values can be represented before wrapping (via masking).
        Increase these if your spectrogram has more than 1024 frequency
        bins, or you expect time deltas beyond 1024 hop-frames.
        """
        freq_mask = (1 << freq_bits) - 1
        time_mask = (1 << time_bits) - 1
        f1 = freq1 & freq_mask
        f2 = freq2 & freq_mask
        dt = delta_t & time_mask
        return (f1 << (freq_bits + time_bits)) | (f2 << time_bits) | dt

 
    def build_fingerprint(self, keypoints, fan_out=5, min_time_delta=1, max_time_delta=100):
        """
        Turn a list of keypoints into a list of (hash, anchor_time) pairs.
 
        Anchor + target scheme: each keypoint acts as an anchor, paired with
        a handful of keypoints that follow it in time (its "target zone").
        Only pairs, not individual peaks, are hashed -- a single peak's
        (time, freq) is common and collides constantly across unrelated
        clips; the *combination* of two peaks' frequencies plus their time
        gap is what makes a hash distinctive enough to identify a specific
        call rather than matching anything with similar-pitched energy.
 
        Parameters
        ----------
        keypoints : list of (time_idx, freq_idx, magnitude)
            Output of extract_keypoints.
        fan_out : int
            Max number of target points paired with each anchor. Main lever
            on fingerprint size and match density: fan_out=1 gives very few,
            very specific hashes (high precision, may miss real matches
            under noise); fan_out=10+ gives more chances to match but more
            spurious collisions. 3-5 is a reasonable start.
        min_time_delta, max_time_delta : int
            Only pair an anchor with targets whose time gap falls in this
            window (in spectrogram time-bins). min_time_delta avoids pairing
            a peak with itself/a near-duplicate; max_time_delta keeps hashes
            tied to *local* call structure, which stays robust when the same
            call appears elsewhere in a mixture with unrelated sounds
            intervening further away in time.
 
        Returns
        -------
        list of (hash, anchor_time) tuples. anchor_time is in the same
        time-bin units as the input keypoints.
        """
        if not keypoints:
            return []
 
        sorted_kps = sorted(keypoints, key=lambda p: p[0])
        n = len(sorted_kps)
        fingerprint = []
 
        for i in range(n):
            t1, f1, _ = sorted_kps[i]
            paired = 0
            for j in range(i + 1, n):
                t2, f2, _ = sorted_kps[j]
                dt = t2 - t1
                if dt < min_time_delta:
                    continue
                if dt > max_time_delta:
                    break  
                h = self._combine_hash(f1, f2, dt)
                fingerprint.append((h, t1))
                paired += 1
                if paired >= fan_out:
                    break
 
        return fingerprint

 
    def add_reference(self, ref_id, spectrogram, **extract_kwargs):
        """
        Compute keypoints + fingerprint for a reference call and merge it
        into this matcher's internal index. Call once per species/sound in
        your reference library.
 
        Parameters
        ----------
        ref_id : hashable
            Identifier for this reference (e.g. species name).
        spectrogram : np.ndarray
            Reference clip's log-magnitude spectrogram.
        **extract_kwargs :
            Passed through to extract_keypoints (neighborhood_size,
            amplitude_floor_db, etc.) if this particular reference needs
            different peak-picking settings than your defaults.
        """
        keypoints = self.extract_keypoints(spectrogram, **extract_kwargs)
        fingerprint = self.build_fingerprint(keypoints, **self.fingerprint_kwargs)
 
        for h, anchor_time in fingerprint:
            self.reference_index.setdefault(h, []).append((ref_id, anchor_time))
        self.reference_hash_counts[ref_id] = (
            self.reference_hash_counts.get(ref_id, 0) + len(fingerprint)
        )

 
    def remove_reference(self, ref_id):
        """Remove a reference's hashes from the index (e.g. to rebuild it with new settings)."""
        for h in list(self.reference_index.keys()):
            self.reference_index[h] = [
                (rid, t) for rid, t in self.reference_index[h] if rid != ref_id
            ]
            if not self.reference_index[h]:
                del self.reference_index[h]
        self.reference_hash_counts.pop(ref_id, None)
 

    def match_and_score(self, query_fingerprint, min_raw_count=1):
        """
        Match a query fingerprint against this matcher's reference index
        and score each reference by how strongly it appears to be present.
 
        Offset-histogram voting: real matches cluster at one consistent
        time offset (the reference clip appears at exactly one point in
        the query); coincidental hash collisions land at random offsets
        and don't accumulate. Counting per-offset votes, rather than total
        hash hits, is what separates true presence from noise.
 
        Returns
        -------
        dict mapping ref_id -> {"score": float, "raw_count": int, "offset": int}
        score is normalized by the reference's own hash count (so short and
        long reference calls are compared fairly); offset is the estimated
        start time of the reference within the query.
        """
        offset_histograms = {}
 
        for h, query_time in query_fingerprint:
            matches = self.reference_index.get(h)
            if not matches:
                continue
            for ref_id, ref_time in matches:
                offset = query_time - ref_time
                bucket = offset_histograms.setdefault(ref_id, {})
                bucket[offset] = bucket.get(offset, 0) + 1
 
        results = {}
        for ref_id, hist in offset_histograms.items():
            best_offset, raw_count = max(hist.items(), key=lambda kv: kv[1])
            if raw_count < min_raw_count:
                continue
            total = self.reference_hash_counts.get(ref_id)
            score = raw_count / total if total else float(raw_count)
            results[ref_id] = {"score": score, "raw_count": raw_count, "offset": best_offset}
 
        return results

 
    @staticmethod
    def rank_matches(results, threshold=0.0, top_k=None):
        """
        Sort match_and_score's output by score, dropping anything below
        `threshold`, capped at `top_k` if given.
 
        Returns a list of (ref_id, info_dict) tuples, highest score first.
        """
        ranked = [
            (ref_id, info) for ref_id, info in results.items() if info["score"] >= threshold
        ]
        ranked.sort(key=lambda kv: kv[1]["score"], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
 

    def detect(self, query_spectrogram, min_raw_count=1, threshold=0.0, top_k=None, **extract_kwargs):
        """
        Full pipeline in one call: extract keypoints -> fingerprint ->
        match against the internal reference index -> rank.
 
        Use this once your reference library is built via add_reference;
        for anything needing intermediate outputs (e.g. inspecting raw
        keypoints for tuning), call the individual methods directly instead.
 
        Returns a list of (ref_id, info_dict) tuples, highest score first --
        empty list if the reference library is empty or nothing matched.
        """
        keypoints = self.extract_keypoints(query_spectrogram, **extract_kwargs)
        fingerprint = self.build_fingerprint(keypoints, **self.fingerprint_kwargs)
        results = self.match_and_score(fingerprint, min_raw_count=min_raw_count)
        return self.rank_matches(results, threshold=threshold, top_k=top_k)
 

    def save(self, path):
        """Pickle the reference index + hash counts + fingerprint settings to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "reference_index": self.reference_index,
                    "reference_hash_counts": self.reference_hash_counts,
                    "fingerprint_kwargs": self.fingerprint_kwargs,
                },
                f,
            )

 
    @classmethod
    def load(cls, path):
        """Load a previously saved reference library into a new AudioMatcher."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        matcher = cls(**state["fingerprint_kwargs"])
        matcher.reference_index = state["reference_index"]
        matcher.reference_hash_counts = state["reference_hash_counts"]
        return matcher