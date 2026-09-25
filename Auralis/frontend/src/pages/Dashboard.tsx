import { useMemo, useRef, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { uploadSample, type UploadSampleResponse } from '@/services/api';
import { Navbar } from '@/components/layout/Navbar';
import { ForestBackground } from '@/components/layout/ForestBackground';
import { AudioPlayerPreview } from '@/components/audio/AudioPlayerPreview';
import { Button } from '@/components/ui/button';
import { Activity, AlertCircle, Award, ChevronDown, ChevronUp, FileAudio, Info, Layers, RefreshCw, Sparkles, TreePine, UploadCloud, X } from 'lucide-react';

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a'];
const FINAL_THRESHOLD = 0.2;
type ModelName = 'dsp' | 'cnn' | 'pann';
interface AnimalResult { species: string; confidence: number; scores: Record<ModelName, number>; }

const MODEL_DETAILS: Record<ModelName, { label: string; weight: number; description: string }> = {
  pann: { label: 'PANN', weight: 0.3, description: 'Pre-trained AudioSet tagging' },
  cnn: { label: 'CNN', weight: 0.4, description: 'Auralis self-trained classifier' },
  dsp: { label: 'DSP', weight: 0.3, description: 'Acoustic fingerprint matcher' },
};
const ANIMAL_EMOJI: Record<string, string> = {
  cat: '🐈', cow: '🐄', crow: '🐦‍⬛', dog: '🐕', frog: '🐸', goat: '🐐', horse: '🐎',
  owl: '🦉', rooster: '🐓', roaring: '🐆', wolf: '🐺', farm_animals: '🐾', gunshot: '🔊',
};

function scoreMap(predictions: { species: string; confidence: number }[]) {
  return new Map(predictions.map(({ species, confidence }) => [species, confidence]));
}

function combinePathResults(response: UploadSampleResponse): AnimalResult[] {
  const pathScores: Record<ModelName, Map<string, number>> = {
    dsp: scoreMap(response.paths.dsp), cnn: scoreMap(response.paths.cnn), pann: scoreMap(response.paths.pann),
  };
  const candidates = new Set([...pathScores.dsp.keys(), ...pathScores.cnn.keys(), ...pathScores.pann.keys()]);
  // Keep every animal nominated above the cutoff by at least one path. The
  // weighted ensemble score is still calculated and ranked, but it must not
  // hide a valid multi-animal candidate merely because the other two paths
  // gave it zero. (For example, CNN 0.50 becomes 0.20 after its 0.4 weight.)
  return [...candidates].map((species) => {
    const scores = { dsp: pathScores.dsp.get(species) ?? 0, cnn: pathScores.cnn.get(species) ?? 0, pann: pathScores.pann.get(species) ?? 0 };
    return { species, scores, confidence: 0.3 * scores.pann + 0.4 * scores.cnn + 0.3 * scores.dsp };
  }).filter((result) => Object.values(result.scores).some((score) => score > FINAL_THRESHOLD))
    .sort((left, right) => right.confidence - left.confidence);
}

function formatSpeciesName(raw: string) {
  return raw.replace(/[_-]/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function confidenceStyle(score: number) {
  if (score >= 0.7) return ['High Confidence', 'text-emerald-300 bg-emerald-500/20 border-emerald-400/30', 'from-emerald-500 to-emerald-300'];
  if (score >= 0.4) return ['Moderate Match', 'text-teal-300 bg-teal-500/20 border-teal-400/30', 'from-teal-600 to-teal-300'];
  return ['Acoustic Trace', 'text-zinc-400 bg-zinc-800/40 border-zinc-700/40', 'from-teal-700 to-emerald-500'];
}

export function Dashboard() {
  const { user } = useAuth();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<UploadSampleResponse | null>(null);
  const [analyzedFileName, setAnalyzedFileName] = useState('');
  const [openDetails, setOpenDetails] = useState<string | null>(null);
  const finalResults = useMemo(() => analysis ? combinePathResults(analysis) : [], [analysis]);
  const pathErrorText = analysis?.path_errors ? Object.entries(analysis.path_errors).map(([name, message]) => `${name.toUpperCase()}: ${message}`).join(' ') : '';

  const validateAndSetFile = (file: File) => {
    setServerError(null);
    const extension = `.${file.name.split('.').pop()?.toLowerCase()}`;
    if (!ALLOWED_EXTENSIONS.includes(extension)) { setServerError(`Invalid file type "${extension}". Please upload an audio file (${ALLOWED_EXTENSIONS.join(', ')}).`); return; }
    if (file.size > 25 * 1024 * 1024) { setServerError('Audio file exceeds maximum size of 25MB.'); return; }
    setSelectedFile(file); setAnalysis(null); setOpenDetails(null);
  };

  const handleProcessAudio = async () => {
    if (!selectedFile) return;
    setIsProcessing(true); setServerError(null);
    const formData = new FormData(); formData.append('sample', selectedFile);
    try { const result = await uploadSample(formData); setAnalysis(result); setAnalyzedFileName(selectedFile.name); setOpenDetails(null); }
    catch (err: unknown) { setServerError(err instanceof Error ? err.message : 'Failed to process audio recording. Please verify the backend service is running and try again.'); }
    finally { setIsProcessing(false); }
  };

  const handleReset = () => { setSelectedFile(null); setAnalysis(null); setOpenDetails(null); setServerError(null); if (fileInputRef.current) fileInputRef.current.value = ''; };

  return <div className="relative min-h-screen w-full bg-[#06140e] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
    <ForestBackground /><Navbar />
    <main className="relative z-10 pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col gap-8">
      <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 sm:p-7 rounded-3xl bg-[#091f16]/90 border border-emerald-500/20 shadow-xl backdrop-blur-xl">
        <div className="flex items-center gap-4"><div className="w-11 h-11 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center"><TreePine className="w-5 h-5 text-emerald-300" /></div><div><h1 className="text-xl sm:text-2xl font-bold font-['Space_Grotesk'] text-white">Bioacoustic Workspace</h1><p className="text-xs sm:text-sm text-emerald-200/70 font-light">Welcome, <span className="font-semibold text-white">{user?.name || 'Ecologist'}</span> • Role: <span className="uppercase font-semibold text-emerald-300">{user?.user_type || 'Client'}</span></p></div></div>
        <span className="inline-flex self-start sm:self-center items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/15 border border-emerald-400/25 text-xs font-mono font-medium text-emerald-200"><span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />Three-path ensemble online</span>
      </section>
      {serverError && <div className="p-4 rounded-2xl bg-red-950/70 border border-red-500/30 flex items-start gap-3 text-sm text-red-200"><AlertCircle className="w-5 h-5 shrink-0 text-red-400 mt-0.5" /><div><h4 className="font-semibold text-red-300">Detection Error</h4><p className="text-xs text-red-300/80 mt-0.5">{serverError}</p></div></div>}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 flex flex-col gap-6"><div className="bg-[#091f16]/90 border border-emerald-500/20 rounded-3xl p-6 shadow-xl backdrop-blur-xl flex flex-col gap-5">
          <div className="flex items-center justify-between pb-3 border-b border-emerald-500/15"><h2 className="text-base font-bold font-['Space_Grotesk'] text-white flex items-center gap-2"><FileAudio className="w-4 h-4 text-emerald-400" />Field Audio Input</h2>{selectedFile && <button onClick={handleReset} className="text-xs text-emerald-300 hover:text-white transition-colors flex items-center gap-1 cursor-pointer font-light"><RefreshCw className="w-3 h-3" />Reset</button>}</div>
          {!selectedFile ? <div onDrop={(event) => { event.preventDefault(); setIsDragging(false); if (event.dataTransfer.files[0]) validateAndSetFile(event.dataTransfer.files[0]); }} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onClick={() => fileInputRef.current?.click()} className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center gap-3 ${isDragging ? 'border-emerald-400 bg-emerald-500/20 scale-[1.01]' : 'border-emerald-500/25 bg-[#0b241a]/50 hover:border-emerald-400/50 hover:bg-[#0e2c20]/70'}`}><input ref={fileInputRef} type="file" accept=".mp3,.wav,.flac,.ogg,.m4a" onChange={(event) => event.target.files?.[0] && validateAndSetFile(event.target.files[0])} className="hidden" /><div className="w-13 h-13 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center text-emerald-300 shadow-md"><UploadCloud className="w-6 h-6" /></div><div><h3 className="text-sm font-semibold text-white">Drop woodland recording here</h3><p className="text-xs text-emerald-200/60 mt-1 font-light">or click to browse local files</p></div><span className="text-[11px] font-mono text-emerald-300/80 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">MP3, WAV, FLAC, OGG, M4A • Max 25MB</span></div> : <div className="flex flex-col gap-4"><AudioPlayerPreview file={selectedFile} onRemove={handleReset} /><Button onClick={handleProcessAudio} disabled={isProcessing} className="w-full h-12 rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold text-sm shadow-xl shadow-emerald-950/40 transition-all duration-300 hover:scale-[1.01] flex items-center justify-center gap-2 cursor-pointer">{isProcessing ? <><span className="w-4 h-4 border-2 border-emerald-950 border-t-transparent rounded-full animate-spin" />Running DSP, CNN & PANN...</> : <><Sparkles className="w-4 h-4" />Run Bioacoustic Identification</>}</Button></div>}
          <div className="p-4 rounded-2xl bg-[#061710]/60 border border-emerald-500/15 text-xs text-zinc-300 flex flex-col gap-2 font-light"><div className="flex items-center gap-1.5 font-medium text-emerald-300"><Info className="w-4 h-4 text-emerald-400" />Field Audio Guidelines</div><ul className="list-disc list-inside space-y-1 text-zinc-300/80 text-[11px] leading-relaxed"><li>Leading and trailing silence are trimmed before DSP matching.</li><li>Audio is normalized for each model's required sample rate and format.</li><li>Solo calls and multi-species recordings are supported.</li></ul></div>
        </div></div>
        <div className="lg:col-span-7 flex flex-col gap-6"><div className="bg-[#091f16]/90 border border-emerald-500/20 rounded-3xl p-6 sm:p-7 shadow-xl backdrop-blur-xl min-h-[480px] flex flex-col">
          <div className="flex items-center justify-between pb-4 border-b border-emerald-500/15 mb-6"><div><h2 className="text-lg font-bold font-['Space_Grotesk'] text-white flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-400" />Species Identification Results</h2><p className="text-xs text-emerald-200/60 mt-0.5 font-light">{analysis ? `Ensemble results for "${analyzedFileName}"` : 'Awaiting audio file submission'}</p></div>{analysis && finalResults.length > 0 && <span className="px-3 py-1 rounded-full bg-emerald-500/15 border border-emerald-400/25 text-xs font-mono font-medium text-emerald-200">{finalResults.length} Species Detected</span>}</div>
          {isProcessing && <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-5"><div className="relative w-20 h-20 rounded-full border border-emerald-500/30 flex items-center justify-center"><div className="absolute inset-0 rounded-full border-2 border-emerald-400 border-t-transparent animate-spin" /><TreePine className="w-8 h-8 text-emerald-300 animate-pulse" /></div><div className="max-w-sm"><h3 className="text-base font-bold text-white font-['Space_Grotesk']">Analyzing with three independent paths</h3><p className="text-xs text-emerald-200/70 mt-1 leading-relaxed font-light">Trimming and normalizing the recording, then running DSP fingerprints, the Auralis CNN, and pre-trained PANN.</p></div></div>}
          {!isProcessing && !analysis && <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4 border border-dashed border-emerald-500/20 rounded-2xl bg-[#061710]/40"><div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-400/20 flex items-center justify-center text-emerald-300"><Layers className="w-7 h-7 opacity-80" /></div><div className="max-w-sm"><h3 className="text-base font-semibold text-white font-['Space_Grotesk']">No Audio Analyzed Yet</h3><p className="text-xs text-zinc-400 mt-1 leading-relaxed font-light">Upload an audio file and run the three-path identification ensemble.</p></div></div>}
          {!isProcessing && analysis && <>{pathErrorText && <div className="mb-4 p-3 rounded-xl bg-amber-950/40 border border-amber-400/20 text-xs text-amber-100 flex gap-2"><Info className="w-4 h-4 shrink-0 text-amber-300" /><span>Partial analysis: {pathErrorText}</span></div>}{finalResults.length > 0 ? <div className="flex flex-col gap-4 animate-in fade-in"><div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-2"><div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15"><span className="text-[11px] text-emerald-200/60 block font-light">Primary Match</span><span className="text-sm font-bold text-white font-['Space_Grotesk'] truncate block">{formatSpeciesName(finalResults[0].species)}</span></div><div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15"><span className="text-[11px] text-emerald-200/60 block font-light">Weighted Confidence</span><span className="text-sm font-mono font-bold text-emerald-300 block">{(finalResults[0].confidence * 100).toFixed(1)}%</span></div><div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15 col-span-2 sm:col-span-1"><span className="text-[11px] text-emerald-200/60 block font-light">Final Threshold</span><span className="text-sm font-mono font-bold text-white block">&gt; {(FINAL_THRESHOLD * 100).toFixed(0)}%</span></div></div>
            {finalResults.map((animal, index) => { const [badgeText, badgeColor, barColor] = confidenceStyle(animal.confidence); const detailOpen = openDetails === animal.species; return <div key={animal.species} className={`p-4 rounded-2xl border transition-all duration-200 ${index === 0 ? 'bg-[#0d2a1f]/80 border-emerald-400/30' : 'bg-[#0b241a]/60 border-emerald-500/15'}`}><div className="flex items-center justify-between gap-3"><div className="flex items-center gap-3 min-w-0"><div className="w-12 h-12 shrink-0 rounded-2xl bg-gradient-to-br from-emerald-400/25 to-teal-800/50 border border-emerald-300/20 flex items-center justify-center text-2xl" aria-label={`${formatSpeciesName(animal.species)} icon`}>{ANIMAL_EMOJI[animal.species] ?? '🐾'}</div><div className="min-w-0"><h4 className="text-base font-semibold text-white font-['Space_Grotesk'] flex items-center gap-2 truncate">{formatSpeciesName(animal.species)}{index === 0 && <Award className="w-4 h-4 shrink-0 text-emerald-400" />}</h4><span className="text-xs text-emerald-300/60 font-mono">ID: {animal.species}</span></div></div><div className="text-right flex flex-col items-end gap-1 shrink-0"><span className="text-base font-mono font-bold text-emerald-300">{(animal.confidence * 100).toFixed(1)}%</span><span className={`text-[10px] font-medium uppercase tracking-wider px-2.5 py-0.5 rounded-full border ${badgeColor}`}>{badgeText}</span></div></div><div className="w-full h-1.5 rounded-full bg-[#06150e] overflow-hidden mt-3"><div className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700`} style={{ width: `${Math.max(5, Math.min(100, animal.confidence * 100))}%` }} /></div><div className="mt-3 flex justify-end"><button type="button" onClick={() => setOpenDetails(detailOpen ? null : animal.species)} className="inline-flex items-center gap-1.5 text-xs text-emerald-300 hover:text-white transition-colors cursor-pointer">{detailOpen ? <><ChevronUp className="w-3.5 h-3.5" />Hide path details</> : <><ChevronDown className="w-3.5 h-3.5" />View path details</>}</button></div>{detailOpen && <div className="relative mt-3 p-3.5 rounded-xl bg-[#061710]/70 border border-emerald-500/15 animate-in fade-in"><button type="button" onClick={() => setOpenDetails(null)} className="absolute right-2.5 top-2.5 p-1 text-emerald-300/70 hover:text-white rounded-md hover:bg-white/5" aria-label="Close path details"><X className="w-4 h-4" /></button><p className="text-[11px] text-emerald-100/70 mb-3 pr-6">Final = 30% PANN + 40% CNN + 30% DSP</p><div className="grid grid-cols-1 sm:grid-cols-3 gap-2">{(Object.keys(MODEL_DETAILS) as ModelName[]).map((model) => <div key={model} className="rounded-lg border border-emerald-500/10 bg-black/10 px-3 py-2"><div className="flex justify-between gap-2 text-xs"><span className="font-semibold text-emerald-200">{MODEL_DETAILS[model].label}</span><span className="font-mono text-white">{(animal.scores[model] * 100).toFixed(1)}%</span></div><p className="mt-1 text-[10px] text-emerald-200/50">{MODEL_DETAILS[model].weight * 100}% weight · {MODEL_DETAILS[model].description}</p></div>)}</div></div>}</div>; })}
          </div> : <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4 border border-dashed border-emerald-500/20 rounded-2xl bg-[#061710]/40"><div className="w-13 h-13 rounded-2xl bg-amber-500/10 border border-amber-400/20 flex items-center justify-center text-amber-300"><Info className="w-6 h-6" /></div><div className="max-w-md"><h3 className="text-base font-semibold text-white font-['Space_Grotesk']">No Final Matches Above 20%</h3><p className="text-xs text-zinc-400 mt-1 leading-relaxed font-light">No candidate exceeded the weighted ensemble threshold. Individual model candidates below 20% are intentionally excluded from the final result.</p></div></div>}</>}
        </div></div>
      </div>
    </main>
  </div>;
}
