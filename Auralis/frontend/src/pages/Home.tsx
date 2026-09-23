import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Navbar } from '@/components/layout/Navbar';
import { ForestVideoBackground } from '@/components/layout/ForestVideoBackground';
import { Button } from '@/components/ui/button';
import { 
  Sparkles, 
  ArrowRight, 
  BookOpen, 
  Radio, 
  Cpu, 
  Leaf, 
  TreePine
} from 'lucide-react';

export function Home() {
  const navigate = useNavigate();
  const { user } = useAuth();

  return (
    <div className="relative min-h-screen w-full bg-[#040d0a] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
      
      <ForestVideoBackground />

      {/* Persistent Navigation */}
      <Navbar />

      {/* Main Content Container */}
      <main className="relative z-10 pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col gap-28">

        {/* HERO SECTION */}
        <section className="min-h-[82vh] flex flex-col justify-center items-center text-center max-w-4xl mx-auto pt-8">

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight font-['Space_Grotesk'] leading-[1.1] mb-6">
            Decoding the <span className="forest-gradient-text">Symphony</span> of the Forest
          </h1>

          {/* Subtitle — intentionally left blank, spacing preserved */}
          <p className="text-base sm:text-xl text-emerald-100/80 max-w-2xl font-light leading-relaxed mb-10">
            &nbsp;
          </p>

          {/* Primary Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full max-w-md mb-16">
            {user ? (
              <Button
                size="lg"
                onClick={() => navigate('/dashboard')}
                className="w-full sm:w-auto px-8 py-6 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold text-base shadow-xl shadow-emerald-500/25 transition-all duration-300 hover:scale-[1.02] flex items-center justify-center gap-2 cursor-pointer"
              >
                Go to Dashboard
                <ArrowRight className="w-5 h-5" />
              </Button>
            ) : (
              <>
                <Button
                  size="lg"
                  onClick={() => navigate('/auth/signup')}
                  className="w-full sm:w-auto px-8 py-6 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold text-base shadow-xl shadow-emerald-500/25 transition-all duration-300 hover:scale-[1.02] flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Sparkles className="w-5 h-5" />
                  Get Started
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => navigate('/auth/login')}
                  className="w-full sm:w-auto px-8 py-6 rounded-xl bg-emerald-950/40 border-emerald-500/30 text-emerald-200 hover:bg-emerald-500/20 hover:text-white font-semibold text-base backdrop-blur-md cursor-pointer"
                >
                  Sign In
                </Button>
              </>
            )}
            <Button
              variant="ghost"
              size="lg"
              onClick={() => navigate('/learn-more')}
              className="w-full sm:w-auto px-6 py-6 rounded-xl text-emerald-300 hover:text-white hover:bg-emerald-500/10 font-medium text-base flex items-center justify-center gap-2 cursor-pointer"
            >
              <BookOpen className="w-4 h-4 text-emerald-400" />
              Learn More
            </Button>
          </div>

        </section>

        {/* FEATURES HIGHLIGHT SECTION */}
        <section className="flex flex-col gap-10">
          <div className="text-center max-w-2xl mx-auto">
            <h2 className="text-2xl sm:text-4xl font-bold font-['Space_Grotesk'] text-white mb-3">
              Built for Ecological Precision
            </h2>
            <p className="text-sm sm:text-base text-emerald-200/70">
              Harnessing non-invasive digital signal processing algorithms to listen where cameras cannot see.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Feature 1 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-8 flex flex-col gap-4 transition-all duration-300 group">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
                <Radio className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white font-['Space_Grotesk']">
                Bandpass & STFT Filtering
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Isolates biological audio between 50 Hz and 10 kHz with Hann-windowed Short-Time Fourier Transforms, eliminating atmospheric distortion and ambient rainfall.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-8 flex flex-col gap-4 transition-all duration-300 group">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white font-['Space_Grotesk']">
                Acoustic Fingerprint Hash
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Keypoint extraction captures the unique harmonic signatures of species calls, cross-referencing against our reference library in sub-seconds.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="forest-glass-card rounded-2xl p-6 sm:p-8 flex flex-col gap-4 transition-all duration-300 group">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
                <TreePine className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white font-['Space_Grotesk']">
                Non-Invasive Monitoring
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed font-light">
                Capture continuous, unbiased field census data without human intrusion, preserving sensitive nests, habitats, and nocturnal wildlife behavior.
              </p>
            </div>

          </div>
        </section>

        {/* BOTTOM CALL TO ACTION BANNER */}
        <section className="forest-glass rounded-3xl p-8 sm:p-12 text-center max-w-4xl mx-auto flex flex-col items-center gap-6 border border-emerald-500/20">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-400">
            <Leaf className="w-6 h-6" />
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold font-['Space_Grotesk'] text-white">
            Ready to listen to the wilderness?
          </h2>
          <p className="text-emerald-100/70 max-w-xl text-sm sm:text-base">
            Upload your field recordings, analyze forest acoustics, and discover which species inhabit the canopy.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
            <Button
              size="lg"
              onClick={() => navigate(user ? '/dashboard' : '/auth/signup')}
              className="px-8 py-6 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold text-base shadow-xl shadow-emerald-500/30 transition-all hover:scale-[1.02] cursor-pointer"
            >
              {user ? 'Enter Dashboard' : 'Create Free Account'}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => navigate('/learn-more')}
              className="px-8 py-6 rounded-xl bg-emerald-950/40 border-emerald-500/30 text-emerald-300 hover:text-white hover:bg-emerald-500/20 text-base cursor-pointer"
            >
              Explore the Science
            </Button>
          </div>
        </section>

      </main>

      {/* MINIMALIST FOREST FOOTER */}
      <footer className="relative z-10 border-t border-emerald-500/15 bg-[#030907]/90 py-8 px-4 sm:px-8">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-emerald-300/60">
          <div className="flex items-center gap-2">
            <TreePine className="w-4 h-4 text-emerald-400" />
            <span className="font-semibold text-zinc-300 font-['Space_Grotesk']">Auralis</span>
            <span>• CSE 2-2 Bioacoustic Term Project</span>
          </div>
          <div className="flex items-center gap-6">
            <span className="font-mono text-[11px]">Spectral Range: 50Hz - 10,000Hz</span>
            <button 
              onClick={() => navigate('/learn-more')}
              className="hover:text-emerald-300 transition-colors cursor-pointer"
            >
              Documentation
            </button>
          </div>
        </div>
      </footer>

    </div>
  );
}
