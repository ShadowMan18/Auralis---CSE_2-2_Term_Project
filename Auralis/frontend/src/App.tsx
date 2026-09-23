import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { LanguageProvider } from './context/LanguageContext';
import { NotificationProvider } from './context/NotificationContext';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './context/ToastContext';
import { RequireGuest } from './components/routes/RequireGuest';
import { RequireAuth } from './components/routes/RequireAuth';
import { RequireRole } from './components/routes/RequireRole';
import { NotFound } from './pages/NotFound';
import { Unauthorized } from './pages/auth/Unauthorized';
import { Signup } from './pages/auth/Signup';
import { Login } from './pages/auth/Login';
import { Logout } from './pages/auth/Logout';
import { Home } from './pages/Home';
import { LearnMore } from './pages/LearnMore';
import { Dashboard } from './pages/Dashboard';

export default function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <LanguageProvider>
          <ToastProvider>
            <NotificationProvider>
              <BrowserRouter>
                <Routes>
                  
                  {/* Public informational & landing routes */}
                  <Route path='/' element={<Home />} />
                  <Route path='/home' element={<Home />} />
                  <Route path='/learn-more' element={<LearnMore />} />

                  {/* Guest-only routes (redirects to /dashboard if logged in) */}
                  <Route element={<RequireGuest />}>
                    <Route path='/auth/signup' element={<Signup />} />
                    <Route path='/auth/login' element={<Login />} />
                  </Route>

                  {/* Protected routes (requires valid login) */}
                  <Route element={<RequireAuth />}>
                    <Route path='/dashboard' element={<Dashboard />} />
                    <Route path='/auth/logout' element={<Logout />} />
                    <Route path='/unauthorized' element={<Unauthorized />} />

                    {/* Role dependent routes */}
                    <Route element={<RequireRole allowed={['admin']} />}>
                    </Route>

                    <Route element={<RequireRole allowed={['client']} />}>
                    </Route>
                  </Route>

                  {/* 404 Catch-all */}
                  <Route path='*' element={<NotFound />} />

                </Routes>
              </BrowserRouter>
            </NotificationProvider>
          </ToastProvider>
        </LanguageProvider>
      </ThemeProvider>
    </AuthProvider>
  );
}