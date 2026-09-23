import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Compass, ArrowLeft } from 'lucide-react';

export function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen w-full bg-[#040d0a] text-zinc-100 flex flex-col justify-center items-center px-4 text-center selection:bg-emerald-500/30 selection:text-emerald-200">
      <div className="max-w-md forest-glass p-8 sm:p-10 rounded-3xl border border-emerald-500/20 shadow-2xl flex flex-col items-center gap-4">
        <div className="w-14 h-14 rounded-2xl bg-emerald-500/15 border border-emerald-400/30 flex items-center justify-center text-emerald-400">
          <Compass className="w-7 h-7 animate-spin duration-1000" />
        </div>
        <span className="text-4xl sm:text-5xl font-bold font-['Space_Grotesk'] forest-gradient-text">
          404
        </span>
        <h1 className="text-xl font-bold text-white">Lost in the Canopy</h1>
        <p className="text-xs sm:text-sm text-emerald-300/70 font-light">
          The trail you are looking for does not exist in this section of the forest.
        </p>
        <Button
          onClick={() => navigate('/')}
          className="mt-2 bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold rounded-xl px-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Return to Trailhead
        </Button>
      </div>
    </div>
  );
}