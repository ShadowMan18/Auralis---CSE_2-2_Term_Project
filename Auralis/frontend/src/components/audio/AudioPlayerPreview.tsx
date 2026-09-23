import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Music, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface AudioPlayerPreviewProps {
  file: File;
  onRemove?: () => void;
}

export function AudioPlayerPreview({ file, onRemove }: AudioPlayerPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration || 0);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = Number(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const restartAudio = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    audioRef.current.play();
    setIsPlaying(true);
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds === 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full rounded-2xl forest-glass-card p-4 sm:p-5 border border-emerald-500/20 backdrop-blur-xl shadow-xl transition-all">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Top Details */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center shrink-0">
            <Music className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="overflow-hidden">
            <h4 className="text-sm font-semibold text-white truncate">{file.name}</h4>
            <p className="text-xs text-emerald-300/70">
              {formatFileSize(file.size)} • {file.type || 'audio/raw'}
            </p>
          </div>
        </div>

        {onRemove && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onRemove}
            className="text-xs text-zinc-400 hover:text-red-400 hover:bg-red-500/10 px-2.5 h-8"
          >
            Change File
          </Button>
        )}
      </div>

      {/* Waveform Visualization Bars */}
      <div className="h-10 my-2 px-2 flex items-center justify-between gap-1 rounded-xl bg-emerald-950/40 border border-emerald-500/10 overflow-hidden">
        {Array.from({ length: 36 }).map((_, idx) => {
          const height = Math.sin(idx * 0.4) * 45 + 50; // varied wave heights
          const delay = (idx % 6) * 0.15;
          return (
            <div
              key={idx}
              className={`w-1 rounded-full transition-all duration-300 ${
                isPlaying
                  ? 'bg-gradient-to-t from-emerald-600 to-emerald-300 soundwave-bar'
                  : 'bg-emerald-800/40'
              }`}
              style={{
                height: `${isPlaying ? height : 20}%`,
                animationDelay: `${delay}s`,
                animationDuration: `${0.8 + (idx % 4) * 0.2}s`,
              }}
            />
          );
        })}
      </div>

      {/* Audio Slider Controls */}
      <div className="flex items-center gap-3 mt-3">
        <Button
          type="button"
          size="icon"
          onClick={togglePlay}
          className="w-10 h-10 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-emerald-950 shrink-0 shadow-lg shadow-emerald-500/25"
        >
          {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
        </Button>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={restartAudio}
          className="w-8 h-8 rounded-lg text-emerald-300 hover:text-white hover:bg-white/5 shrink-0"
          title="Restart from beginning"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>

        <div className="flex-1 flex flex-col gap-1">
          <input
            type="range"
            min={0}
            max={duration || 100}
            step="0.01"
            value={currentTime}
            onChange={handleSeek}
            className="w-full h-1.5 bg-emerald-950 rounded-lg appearance-none cursor-pointer accent-emerald-400 hover:accent-emerald-300 focus:outline-none"
          />
          <div className="flex justify-between text-[11px] font-mono text-emerald-300/60">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={toggleMute}
          className="w-8 h-8 rounded-lg text-emerald-300 hover:text-white hover:bg-white/5 shrink-0"
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </Button>
      </div>
    </div>
  );
}
