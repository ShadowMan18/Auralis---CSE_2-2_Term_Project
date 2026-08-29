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
                  
                  {/* public-only routes */}
                  <Route element={<RequireGuest />}>

                    <Route path='/' element={<Home />} />
                    <Route path='/home' element={<Home />} />
                    <Route path='/auth/signup' element={<Signup />} />
                    <Route path='/auth/login' element={<Login />} />

                  </Route>

                  {/* protected routes */}
                  <Route element={<RequireAuth />}>
                    
                    {/* role independent routes */}
                    <Route path='/auth/logout' element={<Logout />} />
                    <Route path='/unauthorized' element={<Unauthorized />} />
                    <Route path='/dashboard' element={<Dashboard />} />

                    {/* role dependent routes */}
                    <Route element={<RequireRole allowed={['admin']}/>}>

                    </Route>

                    <Route element={<RequireRole allowed={['client']}/>}>
                    
                    </Route>

                  </Route>

                  {/* public routes */}
                  <Route path='*' element={<NotFound />} />

                </Routes>
              </BrowserRouter>
            </NotificationProvider>
          </ToastProvider>
        </LanguageProvider>
      </ThemeProvider>
    </AuthProvider>
  )
}