import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import {
  AudioWaveform,
  Menu,
  X,
  LogOut
} from 'lucide-react';
// Place your logo file at: src/assets/logo.png (matches the existing
// "@/assets/..." alias already used for forestVideo in Home.tsx)
import logoImage from '@/assets/logo.png';

interface NavbarProps {
  // "transparent" (default): floats with no background, for pages that
  // sit on top of the dark video hero (Home, Learn More).
  // "solid": an opaque deep-forest-green bar, for bright/cream pages
  // (Dashboard) where transparent light text wouldn't read.
  variant?: 'transparent' | 'solid';
}

export function Navbar({ variant = 'transparent' }: NavbarProps) {
  const { user } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;
  const isSolid = variant === 'solid';

  return (
    <header
      className={
        isSolid
          ? 'relative z-50 w-full bg-[#173C2A]'
          : 'fixed top-0 left-0 right-0 z-50 px-4 sm:px-8 pt-6'
      }
    >
      <nav
        className={
          isSolid
            ? 'max-w-7xl mx-auto flex items-center justify-between px-4 sm:px-8 py-4'
            : 'max-w-7xl mx-auto flex items-center justify-between'
        }
      >

        {/* Brand Logo */}
        <Link to="/" className="flex items-center gap-3 focus:outline-none">
          <img src={logoImage} alt="Auralis" className="w-9 h-9 object-contain" />
          <span className="text-xl font-bold tracking-tight text-white font-['Space_Grotesk']">
            Auralis
          </span>
        </Link>

        {/* Desktop Navigation Links — plain text, no pill/box behind them */}
        <div className="hidden md:flex items-center gap-8">
          <Link
            to="/"
            className={`text-sm font-medium transition-colors duration-200 ${isActive('/') || isActive('/home')
                ? 'text-emerald-300'
                : 'text-zinc-200 hover:text-white'
              }`}
          >
            Home
          </Link>
          <Link
            to="/learn-more"
            className={`text-sm font-medium transition-colors duration-200 ${isActive('/learn-more')
                ? 'text-emerald-300'
                : 'text-zinc-200 hover:text-white'
              }`}
          >
            Learn More
          </Link>
          {user && (
            <Link
              to="/dashboard"
              className={`text-sm font-medium transition-colors duration-200 ${isActive('/dashboard')
                  ? 'text-emerald-300'
                  : 'text-zinc-200 hover:text-white'
                }`}
            >
              Dashboard
            </Link>
          )}
        </div>

        {/* Right side: everything is plain text EXCEPT one solid pill
            button — same single button style used everywhere, matching
            "Let's Talk" in the reference. */}
        <div className="hidden md:flex items-center gap-6">
          {user ? (
            <>
              <button
                onClick={() => navigate('/auth/logout')}
                className="text-sm font-medium text-zinc-200 hover:text-white transition-colors"
              >
                Sign Out
              </button>
              <Button
                size="sm"
                onClick={() => navigate('/dashboard')}
                className={
                  isSolid
                    ? 'rounded-full px-6 py-5 bg-[#F5F1E7] hover:bg-white text-[#173C2A] font-semibold'
                    : 'rounded-full px-6 py-5 bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold'
                }
              >
                Analyze Audio
              </Button>
            </>
          ) : (
            <>
              <button
                onClick={() => navigate('/auth/login')}
                className="text-sm font-medium text-zinc-200 hover:text-white transition-colors"
              >
                Sign In
              </button>
              <Button
                size="sm"
                onClick={() => navigate('/auth/signup')}
                className={
                  isSolid
                    ? 'rounded-full px-6 py-5 bg-[#F5F1E7] hover:bg-white text-[#173C2A] font-semibold'
                    : 'rounded-full px-6 py-5 bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold'
                }
              >
                Get Started
              </Button>
            </>
          )}
        </div>

        {/* Mobile Hamburger Toggle */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 rounded-lg text-white hover:bg-white/10 focus:outline-none"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </nav>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div
          className={
            isSolid
              ? 'md:hidden mx-4 sm:mx-8 mb-3 rounded-2xl bg-[#0f2a1c] p-5 flex flex-col gap-3 shadow-2xl'
              : 'md:hidden mt-2 max-w-7xl mx-auto rounded-2xl forest-glass p-5 flex flex-col gap-3 shadow-2xl animate-in fade-in slide-in-from-top-3'
          }
        >
          <Link
            to="/"
            onClick={() => setMobileMenuOpen(false)}
            className="px-3 py-2 rounded-lg text-sm text-zinc-200 hover:bg-white/5"
          >
            Home
          </Link>
          <Link
            to="/learn-more"
            onClick={() => setMobileMenuOpen(false)}
            className="px-3 py-2 rounded-lg text-sm text-zinc-200 hover:bg-white/5"
          >
            Learn More
          </Link>
          {user && (
            <Link
              to="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="px-3 py-2 rounded-lg text-sm text-zinc-200 hover:bg-white/5"
            >
              Dashboard
            </Link>
          )}

          <div className="border-t border-white/10 pt-3 mt-1 flex flex-col gap-2">
            {user ? (
              <>
                <Button
                  size="sm"
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigate('/dashboard');
                  }}
                  className={
                    isSolid
                      ? 'w-full rounded-full bg-[#F5F1E7] hover:bg-white text-[#173C2A] font-semibold'
                      : 'w-full rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold'
                  }
                >
                  <AudioWaveform className="w-4 h-4 mr-1.5" />
                  Analyze Audio
                </Button>
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigate('/auth/logout');
                  }}
                  className="w-full text-center py-2 text-sm text-red-300 hover:text-red-200 flex items-center justify-center gap-1.5"
                >
                  <LogOut className="w-4 h-4" />
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Button
                  size="sm"
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigate('/auth/signup');
                  }}
                  className={
                    isSolid
                      ? 'w-full rounded-full bg-[#F5F1E7] hover:bg-white text-[#173C2A] font-semibold'
                      : 'w-full rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold'
                  }
                >
                  Get Started
                </Button>
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigate('/auth/login');
                  }}
                  className="w-full text-center py-2 text-sm text-zinc-200 hover:text-white"
                >
                  Sign In
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}