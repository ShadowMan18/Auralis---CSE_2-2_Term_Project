export function ForestBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#06140e]">
      {/* Deep atmospheric woodland gradients */}
      <div 
        className="absolute inset-0 opacity-80"
        style={{
          background: 'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(20, 68, 48, 0.45), transparent 70%), radial-gradient(ellipse 60% 40% at 85% 100%, rgba(13, 44, 31, 0.35), transparent 60%), radial-gradient(ellipse 50% 50% at 10% 80%, rgba(9, 31, 22, 0.4), transparent 60%)'
        }}
      />

      {/* Subtle Bioacoustic Topographic Contour Lines */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.035]"
        xmlns="http://www.w3.org/2000/svg"
        width="100%"
        height="100%"
      >
        <defs>
          <pattern
            id="contour-pattern"
            width="600"
            height="400"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M0 80 Q 150 40 300 90 T 600 70 M0 160 Q 200 130 350 180 T 600 150 M0 240 Q 120 280 280 230 T 600 250 M0 320 Q 220 300 380 340 T 600 310"
              fill="none"
              stroke="#34d399"
              strokeWidth="1.5"
            />
            <path
              d="M0 40 Q 180 90 320 30 T 600 50 M0 120 Q 140 80 310 130 T 600 110 M0 200 Q 240 170 390 220 T 600 190 M0 280 Q 160 250 330 300 T 600 270 M0 360 Q 210 390 370 340 T 600 370"
              fill="none"
              stroke="#6ee7b7"
              strokeWidth="1"
              strokeDasharray="4 6"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#contour-pattern)" />
      </svg>

      {/* Fine ambient texture grid */}
      <div 
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: 'radial-gradient(#34d399 1px, transparent 1px)',
          backgroundSize: '28px 28px'
        }}
      />
    </div>
  );
}
