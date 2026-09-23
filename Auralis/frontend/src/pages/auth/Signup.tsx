import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
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
import { DatePicker } from '@/components/ui/datepicker';
import { format } from 'date-fns';
import { 
  TreePine, 
  Eye, 
  EyeOff, 
  UserPlus, 
  AlertCircle, 
  Check, 
  Calendar,
  Lock,
  Mail,
  User
} from 'lucide-react';

const signupFormSchema = z.object({
  user_type: z.enum(['admin', 'client'], {
    required_error: 'Please select user type',
  }),
  name: z.string()
    .min(1, 'Please enter name')
    .max(30, 'Name is too long'),
  email: z.string().email('Please enter a valid email address'),
  password: z.string()
    .min(8, 'Password must contain at least 8 characters')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number')
    .regex(/[!@#$%&]/, 'Password must contain at least one special character (!@#$%&)'),
  date_of_birth: z.date({
    required_error: 'Please select your date of birth',
  }),
});

type SignUpFormValues = z.infer<typeof signupFormSchema>;

export function Signup() {
  const navigate = useNavigate();
  const [serverError, setServerError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  const signupForm = useForm<SignUpFormValues>({
    resolver: zodResolver(signupFormSchema),
    defaultValues: { 
      user_type: 'client',
      name: '', 
      email: '', 
      password: '' 
    },
  });

  const passwordValue = signupForm.watch('password') || '';

  // Password requirement check helper
  const checks = {
    length: passwordValue.length >= 8,
    lowercase: /[a-z]/.test(passwordValue),
    uppercase: /[A-Z]/.test(passwordValue),
    number: /[0-9]/.test(passwordValue),
    special: /[!@#$%&]/.test(passwordValue),
  };

  async function onSubmit(values: SignUpFormValues) {
    setServerError(null);

    const signupFormData = new FormData();
    signupFormData.append('user_type', values.user_type);
    signupFormData.append('name', values.name);
    signupFormData.append('email', values.email);
    signupFormData.append('password', values.password);
    signupFormData.append('date_of_birth', format(values.date_of_birth, 'yyyy-MM-dd'));

    try {
      await authAPI.signup(signupFormData);
      signupForm.reset();
      navigate('/auth/login');
    } catch (err: any) {
      setServerError(err?.message || 'Registration failed. Please check your information and try again.');
    }
  }

  return (
    <div className="relative min-h-screen w-full bg-[#06140e] text-zinc-100 overflow-x-hidden selection:bg-emerald-500/30 selection:text-emerald-200">
      
      {/* Static Atmospheric Forest Background */}
      <ForestBackground />

      {/* Floating Home Navbar */}
      <Navbar />

      {/* Main Centered Container */}
      <main className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 py-28 max-w-lg mx-auto">
        <div className="w-full rounded-3xl bg-[#091f16]/90 border border-emerald-500/20 p-8 sm:p-10 shadow-2xl shadow-black/80 backdrop-blur-xl">
          
          {/* Brand Header */}
          <div className="flex flex-col items-center text-center mb-8">
            <div className="w-11 h-11 rounded-2xl bg-emerald-500/15 border border-emerald-400/25 flex items-center justify-center mb-3">
              <TreePine className="w-5 h-5 text-emerald-300" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold font-['Space_Grotesk'] tracking-tight text-white">
              Create an Account
            </h1>
            <p className="text-xs sm:text-sm text-emerald-200/65 mt-1 font-light">
              Join the bioacoustic forest canopy network
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
          <Form {...signupForm}>
            <form onSubmit={signupForm.handleSubmit(onSubmit)} className="space-y-4">
              
              {/* User Type & Name Row */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <FormField
                  control={signupForm.control}
                  name="user_type"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-medium text-emerald-100/90">
                        User Type
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger className="bg-[#0b241a]/80 border-emerald-500/25 text-white rounded-xl focus:ring-1 focus:ring-emerald-400/50 h-11">
                            <SelectValue placeholder="Role" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent className="bg-[#081c14] border-emerald-500/30 text-zinc-100 rounded-xl">
                          <SelectItem value="client" className="hover:bg-emerald-900/50 cursor-pointer">
                            Client
                          </SelectItem>
                          <SelectItem value="admin" className="hover:bg-emerald-900/50 cursor-pointer">
                            Admin
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage className="text-[11px] text-red-400" />
                    </FormItem>
                  )}
                />

                <FormField
                  control={signupForm.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-medium text-emerald-100/90 flex items-center gap-1.5">
                        <User className="w-3.5 h-3.5 text-emerald-400" />
                        Full Name
                      </FormLabel>
                      <FormControl>
                        <Input
                          placeholder="Dr. Jane Foster"
                          {...field}
                          className="bg-[#0b241a]/80 border-emerald-500/25 text-white placeholder:text-emerald-100/30 rounded-xl focus:border-emerald-400 focus:ring-1 focus:ring-emerald-400/50 h-11"
                        />
                      </FormControl>
                      <FormMessage className="text-[11px] text-red-400" />
                    </FormItem>
                  )}
                />
              </div>

              {/* Email */}
              <FormField
                control={signupForm.control}
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
                        placeholder="ecologist@forest.org"
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
                control={signupForm.control}
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
                          placeholder="Min 8 chars, Aa, 1-9, !@#$%&"
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
                    
                    {/* Password Strength Indicators */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 pt-1.5 text-[10px]">
                      <span className={`flex items-center gap-1 ${checks.length ? 'text-emerald-300 font-medium' : 'text-zinc-500'}`}>
                        <Check className="w-3 h-3" /> 8+ Characters
                      </span>
                      <span className={`flex items-center gap-1 ${checks.uppercase && checks.lowercase ? 'text-emerald-300 font-medium' : 'text-zinc-500'}`}>
                        <Check className="w-3 h-3" /> Upper & Lower
                      </span>
                      <span className={`flex items-center gap-1 ${checks.number ? 'text-emerald-300 font-medium' : 'text-zinc-500'}`}>
                        <Check className="w-3 h-3" /> Number (0-9)
                      </span>
                      <span className={`flex items-center gap-1 ${checks.special ? 'text-emerald-300 font-medium' : 'text-zinc-500'}`}>
                        <Check className="w-3 h-3" /> Symbol (!@#$%&)
                      </span>
                    </div>

                    <FormMessage className="text-[11px] text-red-400" />
                  </FormItem>
                )}
              />

              {/* Date of Birth */}
              <FormField
                control={signupForm.control}
                name="date_of_birth"
                render={({ field }) => (
                  <FormItem className="flex flex-col">
                    <FormLabel className="text-xs font-medium text-emerald-100/90 flex items-center gap-1.5">
                      <Calendar className="w-3.5 h-3.5 text-emerald-400" />
                      Date of Birth
                    </FormLabel>
                    <FormControl>
                      <DatePicker 
                        value={field.value} 
                        onChange={field.onChange} 
                      />
                    </FormControl>
                    <FormMessage className="text-[11px] text-red-400" />
                  </FormItem>
                )}
              />

              {/* Submit Button */}
              <Button
                type="submit"
                disabled={signupForm.formState.isSubmitting}
                className="w-full h-11 rounded-full bg-emerald-400 hover:bg-emerald-300 text-emerald-950 font-semibold text-sm shadow-lg shadow-emerald-950/40 transition-all duration-300 hover:scale-[1.01] mt-4 cursor-pointer"
              >
                {signupForm.formState.isSubmitting ? (
                  <div className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-emerald-950 border-t-transparent rounded-full animate-spin" />
                    Creating Account...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <UserPlus className="w-4 h-4" />
                    Complete Registration
                  </div>
                )}
              </Button>
            </form>
          </Form>

          {/* Footer Switch to Sign In */}
          <div className="mt-6 pt-5 border-t border-emerald-500/15 text-center text-xs text-zinc-300/80 font-light">
            Already registered?{' '}
            <Link
              to="/auth/login"
              className="text-emerald-300 hover:text-white font-medium underline underline-offset-4 ml-1"
            >
              Sign in to existing account
            </Link>
          </div>

        </div>
      </main>

    </div>
  );
}