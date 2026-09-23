import { useState, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { uploadSample } from '@/services/api';
import { Navbar } from '@/components/layout/Navbar';
import { ForestBackground } from '@/components/layout/ForestBackground';
import { AudioPlayerPreview } from '@/components/audio/AudioPlayerPreview';
import { Button } from '@/components/ui/button';
import { 
  UploadCloud, 
  Sparkles, 
  Activity, 
  AlertCircle, 
  FileAudio, 
  TreePine, 
  Layers, 
  Award, 
  Info, 
  RefreshCw
} from 'lucide-react';

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a'];

export function Dashboard() {
  const { user } = useAuth();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);

  // Analysis result states
  const [analyzed, setAnalyzed] = useState(false);
  const [species, setSpecies] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number[]>([]);
  const [analyzedFileName, setAnalyzedFileName] = useState<string>('');

  const validateAndSetFile = (file: File) => {
    setServerError(null);
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setServerError(`Invalid file type "${extension}". Please upload an audio file (${ALLOWED_EXTENSIONS.join(', ')}).`);
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setServerError('Audio file exceeds maximum size of 25MB.');
      return;
    }
    setSelectedFile(file);
    setAnalyzed(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleProcessAudio = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setServerError(null);

    const formData = new FormData();
    formData.append('sample', selectedFile);

    try {
      const result = await uploadSample(formData);
      setSpecies(result.species || []);
      setConfidence(result.confidence || []);
      setAnalyzedFileName(selectedFile.name);
      setAnalyzed(true);
    } catch (err: any) {
      setServerError(
        err?.message || 'Failed to process audio recording. Please verify the backend service is running and try again.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalyzed(false);
    setSpecies([]);
    setConfidence([]);
    setServerError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Format species name cleanly
  const formatSpeciesName = (raw: string) => {
    return raw
      .replace(/[_-]/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  return (
    <div className="relative min-h-screen w-full bg-[#06140e] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
      
      {/* Static Atmospheric Forest Background */}
      <ForestBackground />

      {/* Floating Home Navbar */}
      <Navbar />

      <main className="relative z-10 pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col gap-8">
        
        {/* User Greeting & Status Bar */}
        <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 sm:p-7 rounded-3xl bg-[#091f16]/90 border border-emerald-500/20 shadow-xl backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <div className="w-11 h-11 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center">
              <TreePine className="w-5 h-5 text-emerald-300" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold font-['Space_Grotesk'] text-white">
                Bioacoustic Workspace
              </h1>
              <p className="text-xs sm:text-sm text-emerald-200/70 font-light">
                Welcome, <span className="font-semibold text-white">{user?.name || 'Ecologist'}</span> • Role: <span className="uppercase font-semibold text-emerald-300">{user?.user_type || 'Client'}</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-center">
            <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/15 border border-emerald-400/25 text-xs font-mono font-medium text-emerald-200">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              DSP Engine Online • 22.05 kHz
            </span>
          </div>
        </section>

        {/* Server Error Alert */}
        {serverError && (
          <div className="p-4 rounded-2xl bg-red-950/70 border border-red-500/30 flex items-start gap-3 text-sm text-red-200 animate-in fade-in">
            <AlertCircle className="w-5 h-5 shrink-0 text-red-400 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-red-300">Detection Error</h4>
              <p className="text-xs text-red-300/80 mt-0.5">{serverError}</p>
            </div>
          </div>
        )}

        {/* Dashboard Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT COLUMN: Audio Upload & Player (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            
            <div className="bg-[#091f16]/90 border border-emerald-500/20 rounded-3xl p-6 shadow-xl backdrop-blur-xl flex flex-col gap-5">
              <div className="flex items-center justify-between pb-3 border-b border-emerald-500/15">
                <h2 className="text-base font-bold font-['Space_Grotesk'] text-white flex items-center gap-2">
                  <FileAudio className="w-4 h-4 text-emerald-400" />
                  Field Audio Input
                </h2>
                {selectedFile && (
                  <button
                    onClick={handleReset}
                    className="text-xs text-emerald-300 hover:text-white transition-colors flex items-center gap-1 cursor-pointer font-light"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Reset
                  </button>
                )}
              </div>

              {/* Upload Dropzone */}
              {!selectedFile ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center gap-3 ${
                    isDragging
                      ? 'border-emerald-400 bg-emerald-500/20 scale-[1.01]'
                      : 'border-emerald-500/25 bg-[#0b241a]/50 hover:border-emerald-400/50 hover:bg-[#0e2c20]/70'
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp3,.wav,.flac,.ogg,.m4a"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div className="w-13 h-13 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center text-emerald-300 shadow-md">
                    <UploadCloud className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      Drop woodland recording here
                    </h3>
                    <p className="text-xs text-emerald-200/60 mt-1 font-light">
                      or click to browse local files
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-emerald-300/80 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                    MP3, WAV, FLAC, OGG, M4A • Max 25MB
                  </span>
                </div>
              ) : (
                /* Selected File with Audio Player Preview */
                <div className="flex flex-col gap-4">
                  <AudioPlayerPreview
                    file={selectedFile}
                    onRemove={handleReset}
                  />

                  <Button
                    onClick={handleProcessAudio}
                    disabled={isProcessing}
                    className="w-full h-12 rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold text-sm shadow-xl shadow-emerald-950/40 transition-all duration-300 hover:scale-[1.01] flex items-center justify-center gap-2 cursor-pointer"
                  >
                    {isProcessing ? (
                      <>
                        <span className="w-4 h-4 border-2 border-emerald-950 border-t-transparent rounded-full animate-spin" />
                        Analyzing Soundscape Spectra...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        Run Bioacoustic Identification
                      </>
                    )}
                  </Button>
                </div>
              )}

              {/* Acoustic Recording Guidelines */}
              <div className="p-4 rounded-2xl bg-[#061710]/60 border border-emerald-500/15 text-xs text-zinc-300 flex flex-col gap-2 font-light">
                <div className="flex items-center gap-1.5 font-medium text-emerald-300">
                  <Info className="w-4 h-4 text-emerald-400" />
                  Field Audio Guidelines
                </div>
                <ul className="list-disc list-inside space-y-1 text-zinc-300/80 text-[11px] leading-relaxed">
                  <li>Aim for dawn chorus (5:00 - 8:00 AM) or dusk for peak biophony.</li>
                  <li>Our bandpass filter isolates signals between 50 Hz and 10 kHz.</li>
                  <li>Both solo calls and complex multi-species recordings are supported.</li>
                </ul>
              </div>
            </div>

          </div>

          {/* RIGHT COLUMN: Identification Results (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            
            <div className="bg-[#091f16]/90 border border-emerald-500/20 rounded-3xl p-6 sm:p-7 shadow-xl backdrop-blur-xl min-h-[480px] flex flex-col">
              
              <div className="flex items-center justify-between pb-4 border-b border-emerald-500/15 mb-6">
                <div>
                  <h2 className="text-lg font-bold font-['Space_Grotesk'] text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    Species Identification Results
                  </h2>
                  <p className="text-xs text-emerald-200/60 mt-0.5 font-light">
                    {analyzed ? `Analysis results for "${analyzedFileName}"` : 'Awaiting audio file submission'}
                  </p>
                </div>

                {analyzed && species.length > 0 && (
                  <span className="px-3 py-1 rounded-full bg-emerald-500/15 border border-emerald-400/25 text-xs font-mono font-medium text-emerald-200">
                    {species.length} Species Detected
                  </span>
                )}
              </div>

              {/* PROCESSING LOADING STATE */}
              {isProcessing && (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-5 animate-in fade-in">
                  <div className="relative w-20 h-20 rounded-full border border-emerald-500/30 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border-2 border-emerald-400 border-t-transparent animate-spin" />
                    <TreePine className="w-8 h-8 text-emerald-300 animate-pulse" />
                  </div>
                  <div className="max-w-sm">
                    <h3 className="text-base font-bold text-white font-['Space_Grotesk']">
                      Deconvolving Soundscape Frequencies
                    </h3>
                    <p className="text-xs text-emerald-200/70 mt-1 leading-relaxed font-light">
                      Executing 50Hz-10kHz bandpass isolation, computing Hann STFT spectrograms, and matching against reference acoustic keypoints...
                    </p>
                  </div>
                </div>
              )}

              {/* INITIAL EMPTY STATE (Before upload) */}
              {!isProcessing && !analyzed && (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4 border border-dashed border-emerald-500/20 rounded-2xl bg-[#061710]/40">
                  <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-400/20 flex items-center justify-center text-emerald-300">
                    <Layers className="w-7 h-7 opacity-80" />
                  </div>
                  <div className="max-w-sm">
                    <h3 className="text-base font-semibold text-white font-['Space_Grotesk']">
                      No Audio Analyzed Yet
                    </h3>
                    <p className="text-xs text-zinc-400 mt-1 leading-relaxed font-light">
                      Upload an audio file on the left panel and click &ldquo;Run Bioacoustic Identification&rdquo; to process species vocalizations.
                    </p>
                  </div>
                </div>
              )}

              {/* RESULTS LIST: SPECIES DETECTED */}
              {!isProcessing && analyzed && species.length > 0 && (
                <div className="flex flex-col gap-4 animate-in fade-in">
                  
                  {/* Summary Metric Banner */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-2">
                    <div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15">
                      <span className="text-[11px] text-emerald-200/60 block font-light">Primary Match</span>
                      <span className="text-sm font-bold text-white font-['Space_Grotesk'] truncate block">
                        {formatSpeciesName(species[0])}
                      </span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15">
                      <span className="text-[11px] text-emerald-200/60 block font-light">Top Confidence</span>
                      <span className="text-sm font-mono font-bold text-emerald-300 block">
                        {(confidence[0] * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-[#0b241a]/50 border border-emerald-500/15 col-span-2 sm:col-span-1">
                      <span className="text-[11px] text-emerald-200/60 block font-light">Total Signatures</span>
                      <span className="text-sm font-mono font-bold text-white block">
                        {species.length} detected
                      </span>
                    </div>
                  </div>

                  {/* Species Ranked Cards */}
                  <div className="flex flex-col gap-3">
                    {species.map((sp, idx) => {
                      const score = confidence[idx] !== undefined ? confidence[idx] : 0;
                      const percentage = (score * 100).toFixed(1);
                      const isTop = idx === 0;

                      // Confidence categorization badge
                      let badgeText = 'Acoustic Trace';
                      let badgeColor = 'text-zinc-400 bg-zinc-800/40 border-zinc-700/40';
                      let barColor = 'from-teal-600 to-emerald-400';

                      if (score >= 0.7) {
                        badgeText = 'High Confidence';
                        badgeColor = 'text-emerald-300 bg-emerald-500/20 border-emerald-400/30';
                        barColor = 'from-emerald-500 to-emerald-300';
                      } else if (score >= 0.4) {
                        badgeText = 'Moderate Match';
                        badgeColor = 'text-teal-300 bg-teal-500/20 border-teal-400/30';
                        barColor = 'from-teal-600 to-teal-300';
                      }

                      return (
                        <div
                          key={idx}
                          className={`p-4 rounded-2xl border transition-all duration-200 flex flex-col gap-3 ${
                            isTop 
                              ? 'bg-[#0d2a1f]/80 border-emerald-400/30' 
                              : 'bg-[#0b241a]/60 border-emerald-500/15'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-3">
                              <span className={`w-7 h-7 rounded-xl flex items-center justify-center font-mono text-xs font-semibold ${
                                isTop 
                                  ? 'bg-emerald-400 text-emerald-950 shadow-md shadow-emerald-500/20' 
                                  : 'bg-[#061710] text-emerald-300 border border-emerald-500/20'
                              }`}>
                                #{idx + 1}
                              </span>
                              <div>
                                <h4 className="text-base font-semibold text-white font-['Space_Grotesk'] flex items-center gap-2">
                                  {formatSpeciesName(sp)}
                                  {isTop && (
                                    <Award className="w-4 h-4 text-emerald-400" />
                                  )}
                                </h4>
                                <span className="text-xs text-emerald-300/60 font-mono">
                                  ID: {sp}
                                </span>
                              </div>
                            </div>

                            <div className="text-right flex flex-col items-end gap-1">
                              <span className="text-base font-mono font-bold text-emerald-300">
                                {percentage}%
                              </span>
                              <span className={`text-[10px] font-medium uppercase tracking-wider px-2.5 py-0.5 rounded-full border ${badgeColor}`}>
                                {badgeText}
                              </span>
                            </div>
                          </div>

                          {/* Progress Meter Bar */}
                          <div className="w-full h-1.5 rounded-full bg-[#06150e] overflow-hidden">
                            <div
                              className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700`}
                              style={{ width: `${Math.min(100, Math.max(5, score * 100))}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                </div>
              )}

              {/* ZERO DETECTIONS STATE */}
              {!isProcessing && analyzed && species.length === 0 && (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4 border border-dashed border-emerald-500/20 rounded-2xl bg-[#061710]/40">
                  <div className="w-13 h-13 rounded-2xl bg-amber-500/10 border border-amber-400/20 flex items-center justify-center text-amber-300">
                    <Info className="w-6 h-6" />
                  </div>
                  <div className="max-w-md">
                    <h3 className="text-base font-semibold text-white font-['Space_Grotesk']">
                      No High-Confidence Signatures Found
                    </h3>
                    <p className="text-xs text-zinc-400 mt-1 leading-relaxed font-light">
                      The acoustic matcher did not detect any reference species above the threshold in this audio clip. Ensure the file contains clear wildlife calls within the 50 Hz – 10 kHz band.
                    </p>
                  </div>
                </div>
              )}

            </div>

          </div>

        </div>

      </main>

    </div>
  );
}