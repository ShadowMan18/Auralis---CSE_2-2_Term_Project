import { useNavigate } from 'react-router-dom';
import { Navbar } from '@/components/layout/Navbar';
import { ForestVideoBackground } from '@/components/layout/ForestVideoBackground';
import { Button } from '@/components/ui/button';
import { 
  TreePine, 
  ArrowLeft, 
  Sliders, 
  Cpu, 
  Radio, 
  Microscope, 
  Database, 
  Sparkles,
  CheckCircle2
} from 'lucide-react';

export function LearnMore() {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen w-full bg-[#040d0a] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
      
      <ForestVideoBackground />

      <Navbar />

      <main className="relative z-10 pt-32 pb-24 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto flex flex-col gap-16">
        
        {/* Header Section */}
        <section className="flex flex-col gap-6 text-center max-w-3xl mx-auto">
          <button
            onClick={() => navigate('/')}
            className="inline-flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 self-center transition-colors mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </button>

          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full forest-glass border border-emerald-400/30 self-center text-xs font-semibold text-emerald-300 uppercase tracking-widest">
            Auralis
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold font-['Space_Grotesk'] tracking-tight leading-tight">
            How Bioacoustic <span className="forest-gradient-text">Forest Recordings</span> identifies species
          </h1>

          <p className="text-base sm:text-lg text-emerald-100/75 leading-relaxed font-light">
            Traditional wildlife surveys rely on cameras or intrusive manual field observation.
            Auralis leverages passive acoustic monitoring (PAM) and customized digital signal processing (DSP) to identify species by their unique vocal fingerprints.
          </p>
        </section>

        {/* The 4-Stage DSP Architecture */}
        <section className="flex flex-col gap-8">
          <div className="flex items-center gap-3 border-b border-emerald-500/20 pb-4">
            <Cpu className="w-6 h-6 text-emerald-400" />
            <h2 className="text-2xl font-bold font-['Space_Grotesk'] text-white">
              The Auralis DSP Detection Pipeline
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Stage 1 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-7 flex flex-col gap-3 border border-emerald-500/20">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-emerald-400 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
                  STAGE 01
                </span>
                <Sliders className="w-5 h-5 text-emerald-300/60" />
              </div>
              <h3 className="text-xl font-bold text-white font-['Space_Grotesk']">
                Acoustic Normalization & Preprocessing
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Uploaded field audio (MP3, WAV, FLAC, OGG, M4A) is resampled to a standardized <span className="text-emerald-300 font-mono">22,050 Hz</span> mono waveform. Amplitude ranges are scaled between -1.0 and 1.0 to eliminate gain discrepancies across different microphone models.
              </p>
            </div>

            {/* Stage 2 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-7 flex flex-col gap-3 border border-emerald-500/20">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-emerald-400 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
                  STAGE 02
                </span>
                <Radio className="w-5 h-5 text-emerald-300/60" />
              </div>
              <h3 className="text-xl font-bold text-white font-['Space_Grotesk']">
                50Hz - 10kHz Bandpass Filter
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Forest recordings are plagued by low-frequency wind turbulence and ultra-high frequency electrical hiss. Our customized bandpass filter enforces a 50 Hz lower cutoff and 10,000 Hz upper cutoff with a 200 Hz transition bandwidth to isolate avian and mammalian bio-frequencies.
              </p>
            </div>

            {/* Stage 3 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-7 flex flex-col gap-3 border border-emerald-500/20">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-emerald-400 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
                  STAGE 03
                </span>
                <Microscope className="w-5 h-5 text-emerald-300/60" />
              </div>
              <h3 className="text-xl font-bold text-white font-['Space_Grotesk']">
                Hann-Windowed STFT Spectrograms
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                We compute Short-Time Fourier Transforms using a 46ms window length and 12ms hop length with a Hann window. This yields a complex log-magnitude time-frequency spectrogram capturing overtone structures and pitch modulations.
              </p>
            </div>

            {/* Stage 4 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-7 flex flex-col gap-3 border border-emerald-500/20">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-emerald-400 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
                  STAGE 04
                </span>
                <Database className="w-5 h-5 text-emerald-300/60" />
              </div>
              <h3 className="text-xl font-bold text-white font-['Space_Grotesk']">
                Reference Fingerprint Hash Matching
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Keypoints are extracted and hashed against reference acoustic signatures in memory. The matcher ranks candidate species and outputs normalized confidence scores for immediate review on your dashboard.
              </p>
            </div>

          </div>
        </section>

        {/* DSP Technical Parameters Table */}
        <section className="forest-glass rounded-3xl p-6 sm:p-8 border border-emerald-500/20">
          <div className="flex items-center gap-3 mb-6">
            <TreePine className="w-6 h-6 text-emerald-400" />
            <h3 className="text-xl font-bold font-['Space_Grotesk'] text-white">
              Signal Processing Specification
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-emerald-500/20 text-xs font-mono uppercase text-emerald-300">
                <tr>
                  <th className="py-3 px-4">Parameter</th>
                  <th className="py-3 px-4">Specification Value</th>
                  <th className="py-3 px-4">Engineering Purpose</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-emerald-500/10 text-zinc-300 font-light">
                <tr>
                  <td className="py-3 px-4 font-mono font-medium text-emerald-200">Sample Rate</td>
                  <td className="py-3 px-4 font-mono text-white">22,050 Hz</td>
                  <td className="py-3 px-4">Captures up to 11 kHz Nyquist frequency, covering avian chirps and mammals.</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-mono font-medium text-emerald-200">Bandpass Filter</td>
                  <td className="py-3 px-4 font-mono text-white">50 Hz – 10,000 Hz</td>
                  <td className="py-3 px-4">Suppresses ground vibration, heavy wind turbulence, and electrical noise.</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-mono font-medium text-emerald-200">STFT Window</td>
                  <td className="py-3 px-4 font-mono text-white">46 ms (Hann)</td>
                  <td className="py-3 px-4">Optimized trade-off between temporal resolution and frequency bin selectivity.</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-mono font-medium text-emerald-200">Hop Duration</td>
                  <td className="py-3 px-4 font-mono text-white">12 ms</td>
                  <td className="py-3 px-4">Maintains dense spectral coverage for transient bird calls and quick chirps.</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-mono font-medium text-emerald-200">Supported Ingestion</td>
                  <td className="py-3 px-4 font-mono text-white">.mp3, .wav, .flac, .ogg, .m4a</td>
                  <td className="py-3 px-4">Compatible with diverse field audio capture equipment and handheld recorders.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Conservation Impact */}
        <section className="forest-glass-card rounded-3xl p-8 sm:p-10 border border-emerald-500/20 flex flex-col gap-6">
          <h3 className="text-2xl font-bold font-['Space_Grotesk'] text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            Ecological & Conservation Impact
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-zinc-300 leading-relaxed font-light">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <span><strong>Non-Invasive Surveillance:</strong> Monitors endangered or shy animals without human disruption, vehicle disturbance, or flash photography.</span>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <span><strong>Nocturnal Insights:</strong> Tracks nocturnal raptors, owls, and bats that are virtually impossible to survey via daylight visual checks.</span>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <span><strong>Continuous Habitat Audits:</strong> Detects changes in bioacoustic density, signaling seasonal migration anomalies or habitat fragmentation.</span>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <span><strong>Scalable Data Pipelines:</strong> Processes gigabytes of automated soundscape data within minutes rather than weeks of manual listening.</span>
            </div>
          </div>

          <div className="pt-4 flex flex-wrap gap-4 items-center">
            <Button
              size="lg"
              onClick={() => navigate('/dashboard')}
              className="bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold px-8 shadow-lg shadow-emerald-500/20"
            >
              Analyze Your Audio Now
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => navigate('/auth/signup')}
              className="bg-emerald-950/40 border-emerald-500/30 text-emerald-300 hover:text-white"
            >
              Sign Up
            </Button>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-emerald-500/15 bg-[#030907]/90 py-6 px-4 text-center text-xs text-emerald-300/60">
        Auralis • Automated Forest Bioacoustic Platform
      </footer>

    </div>
  );
}