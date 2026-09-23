import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import * as authAPI from '@/services/auth_api';
import { Navbar } from '@/components/layout/Navbar';
import { ForestBackground } from '@/components/layout/ForestBackground';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { 
  TreePine, 
  Eye, 
  EyeOff, 
  LogIn, 
  AlertCircle, 
  Lock,
  Mail,
  UserCheck
} from 'lucide-react';

const loginFormSchema = z.object({
  user_type: z.enum(['admin', 'client'], {
    required_error: 'Please select your user type',
  }),
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Please enter your password'),
});

type LoginFormValues = z.infer<typeof loginFormSchema>;

export function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [serverError, setServerError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  const loginForm = useForm<LoginFormValues>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: { 
      user_type: 'client',
      email: '', 
      password: '' 
    },
  });

  async function onSubmit(values: LoginFormValues) {
    setServerError(null);

    const formData = new FormData();
    formData.append('user_type', values.user_type);
    formData.append('email', values.email);
    formData.append('password', values.password);

    try {
      const result = await authAPI.login(formData);

      login(
        {
          user_type: values.user_type,
          user_id: result.user_id,
          name: result.name,
          email: values.email,
          profile_picture: result.profile_picture || null,
        },
        result.token
      );

      loginForm.reset();
      navigate('/dashboard');
    } catch (err: any) {
      setServerError(err?.message || 'Invalid credentials or connection error. Please try again.');
    }
  }

  return (
    <div className="relative min-h-screen w-full bg-[#06140e] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
      
      {/* Static Atmospheric Forest Background */}
      <ForestBackground />

      {/* Floating Home Navbar */}
      <Navbar />

      {/* Main Centered Container */}
      <main className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 py-28 max-w-md mx-auto">
        <div className="w-full rounded-3xl bg-[#091f16]/90 border border-emerald-500/20 p-8 sm:p-10 shadow-2xl shadow-black/80 backdrop-blur-xl">
          
          {/* Brand Header */}
          <div className="flex flex-col items-center text-center mb-8">
            <div className="w-11 h-11 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center mb-3">
              <TreePine className="w-5 h-5 text-emerald-300" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold font-['Space_Grotesk'] tracking-tight text-white">
              Sign In
            </h1>
            <p className="text-xs sm:text-sm text-emerald-200/65 mt-1 font-light">
              Access your forest bioacoustic workspace
            </p>
          </div>

          {/* Server Error Alert */}
          {serverError && (
            <div className="mb-6 p-3.5 rounded-xl bg-red-950/70 border border-red-500/30 flex items-start gap-2.5 text-xs text-red-200 animate-in fade-in">
              <AlertCircle className="w-4 h-4 shrink-0 text-red-400 mt-0.5" />
              <span className="flex-1">{serverError}</span>
            </div>
          )}

          {/* Form Container */}
          <Form {...loginForm}>
            <form onSubmit={loginForm.handleSubmit(onSubmit)} className="space-y-5">
              
              {/* User Type */}
              <FormField
                control={loginForm.control}
                name="user_type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs font-medium text-emerald-100/90 flex items-center gap-1.5">
                      <UserCheck className="w-3.5 h-3.5 text-emerald-400" />
                      Account Role
                    </FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger className="bg-[#0b241a]/80 border-emerald-500/25 text-white rounded-xl focus:ring-1 focus:ring-emerald-400/50 h-11">
                          <SelectValue placeholder="Select account type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent className="bg-[#081c14] border-emerald-500/30 text-zinc-100 rounded-xl">
                        <SelectItem value="client" className="hover:bg-emerald-900/50 cursor-pointer">
                          Client / Field Researcher
                        </SelectItem>
                        <SelectItem value="admin" className="hover:bg-emerald-900/50 cursor-pointer">
                          Administrator
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage className="text-[11px] text-red-400" />
                  </FormItem>
                )}
              />

              {/* Email */}
              <FormField
                control={loginForm.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs font-medium text-emerald-100/90 flex items-center gap-1.5">
                      <Mail className="w-3.5 h-3.5 text-emerald-400" />
                      Email Address
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="email"
                        placeholder="researcher@auralis.bio"
                        {...field}
                        className="bg-[#0b241a]/80 border-emerald-500/25 text-white placeholder:text-emerald-100/30 rounded-xl focus:border-emerald-400 focus:ring-1 focus:ring-emerald-400/50 h-11"
                      />
                    </FormControl>
                    <FormMessage className="text-[11px] text-red-400" />
                  </FormItem>
                )}
              />

              {/* Password */}
              <FormField
                control={loginForm.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs font-medium text-emerald-100/90 flex items-center gap-1.5">
                      <Lock className="w-3.5 h-3.5 text-emerald-400" />
                      Password
                    </FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Input
                          type={showPassword ? 'text' : 'password'}
                          placeholder="••••••••••••"
                          {...field}
                          className="bg-[#0b241a]/80 border-emerald-500/25 text-white placeholder:text-emerald-100/30 rounded-xl focus:border-emerald-400 focus:ring-1 focus:ring-emerald-400/50 pr-10 h-11"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-emerald-300/70 hover:text-white focus:outline-none cursor-pointer"
                        >
                          {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </FormControl>
                    <FormMessage className="text-[11px] text-red-400" />
                  </FormItem>
                )}
              />

              {/* Submit Button */}
              <Button
                type="submit"
                disabled={loginForm.formState.isSubmitting}
                className="w-full h-11 rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold text-sm shadow-lg shadow-emerald-950/40 transition-all duration-300 hover:scale-[1.01] mt-2 cursor-pointer"
              >
                {loginForm.formState.isSubmitting ? (
                  <div className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-emerald-950 border-t-transparent rounded-full animate-spin" />
                    Signing in...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </div>
                )}
              </Button>
            </form>
          </Form>

          {/* Footer Switch to Sign Up */}
          <div className="mt-8 pt-6 border-t border-emerald-500/15 text-center text-xs text-zinc-300/80 font-light">
            Don&apos;t have an account?{' '}
            <Link
              to="/auth/signup"
              className="text-emerald-300 hover:text-white font-medium underline underline-offset-4 ml-1"
            >
              Sign up
            </Link>
          </div>

        </div>
      </main>

    </div>
  );
}