import { useRef, useEffect, useState } from 'react';
import forestVideo from '@/assets/forest_video.mp4';

// How many seconds, right at the start/end of the clip, each video fades
// through. Raise this if the handoff is still noticeable; lower it if the
// fade itself starts to look like a fade.
const LOOP_FADE_SECONDS = 1.2;

// Flat, even darkening over the whole video — no gradient, so darkness is
// identical across the whole frame. Raise for darker, lower for brighter.
const VIDEO_DARKNESS = 0.6;

/**
 * Full-page background video with a seamless loop.
 *
 * Two copies of the same clip play simultaneously, offset by half the
 * clip's length. Whichever copy is near ITS OWN start/end (about to loop)
 * fades down, while the other — mid-playthrough — is at full opacity and
 * carries the scene. The two are never near an edge at the same time, so
 * there's always unbroken motion visible and the loop point of each is
 * masked by the other rather than by a dip to black.
 *
 * Drop this once near the top of any full-page layout, before the page's
 * own content, e.g.:
 *   <div className="relative min-h-screen ...">
 *     <ForestVideoBackground />
 *     <Navbar />
 *     <main className="relative z-10 ...">...</main>
 *   </div>
 */
export function ForestVideoBackground() {
  const videoARef = useRef<HTMLVideoElement | null>(null);
  const videoBRef = useRef<HTMLVideoElement | null>(null);
  const [opacityA, setOpacityA] = useState(1);
  const [opacityB, setOpacityB] = useState(0);

  useEffect(() => {
    const a = videoARef.current;
    const b = videoBRef.current;
    if (!a || !b) return;

    let rafId: number | null = null;
    let cancelled = false;
    let started = false;

    const tryStart = () => {
      if (started || cancelled) return;
      if (!a.duration || !b.duration) return;
      started = true;
      b.currentTime = a.duration / 2;
      a.play().catch(() => { });
      b.play().catch(() => { });

      const tick = () => {
        if (a.duration && b.duration) {
          const aEdge = Math.min(a.currentTime, a.duration - a.currentTime);
          const bEdge = Math.min(b.currentTime, b.duration - b.currentTime);
          const oa = aEdge < LOOP_FADE_SECONDS ? aEdge / LOOP_FADE_SECONDS : 1;
          const ob = bEdge < LOOP_FADE_SECONDS ? bEdge / LOOP_FADE_SECONDS : 1;
          setOpacityA(Math.max(0, Math.min(1, oa)));
          setOpacityB(Math.max(0, Math.min(1, ob)));
        }
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    };

    a.addEventListener('loadedmetadata', tryStart);
    b.addEventListener('loadedmetadata', tryStart);
    // In case metadata was already available by the time this effect ran.
    tryStart();

    return () => {
      cancelled = true;
      a.removeEventListener('loadedmetadata', tryStart);
      b.removeEventListener('loadedmetadata', tryStart);
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div className="video-backdrop-wrapper">
      <video
        ref={videoARef}
        src={forestVideo}
        muted
        loop
        playsInline
        className="video-backdrop-video"
        style={{ opacity: opacityA }}
      />
      <video
        ref={videoBRef}
        src={forestVideo}
        muted
        loop
        playsInline
        className="video-backdrop-video"
        style={{ opacity: opacityB }}
      />
      {/* Even darkening overlay — a flat color, not a gradient, so
          darkness is identical across the whole frame. */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ backgroundColor: `rgba(2, 10, 7, ${VIDEO_DARKNESS})` }}
      />
    </div>
  );
}