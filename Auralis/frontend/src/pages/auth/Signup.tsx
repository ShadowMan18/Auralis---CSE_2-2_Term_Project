import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import * as authAPI from '@/services/auth_api';
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
import { MediaPicker } from '@/components/ui/mediapicker';
import { uploadFile } from '@/services/cloud_api/s3';

const signupFormSchema = z.object({
    user_type: z.enum(['admin', 'client'], {
        required_error: 'Please select user type'
    }),
    name: z.string()
        .min(1, 'Please enter name')
        .max(30, 'Name is too long'),
    email: z.string().email(),
    password: z.string()
        .min(1, 'Please enter password')
        .min(8, 'Password must contain at least 8 characters')
        .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
        .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
        .regex(/[0-9]/, 'Password must contain at least one number')
        .regex(/[!@#$%&]/, 'Password must contain at least one special character'),
    date_of_birth: z.date(),
    profile_picture: z.instanceof(File).optional()
});

type SignUpFormValues = z.infer<typeof signupFormSchema>;

export function Signup() {
    const navigate = useNavigate();
    const [serverError, setServerError] = useState<string | null>(null);

    const signupForm = useForm<SignUpFormValues>({
        resolver: zodResolver(signupFormSchema),
        defaultValues: { name: '', email: '', password: '' },
    })

    async function onSubmit(values: SignUpFormValues) {
        setServerError(null);

        const signupFormData = new FormData();
        signupFormData.append('user_type', values.user_type);
        signupFormData.append('name', values.name);
        signupFormData.append('email', values.email)
        signupFormData.append('password', values.password);
        signupFormData.append('date_of_birth', format(values.date_of_birth, 'yyyy-MM-dd'));

        try {
            const result = await authAPI.signup(signupFormData);

            localStorage.setItem('token', result.token);
            
            // onboarding logics
            if (values.profile_picture) {
                const onboardingFormData = new FormData();

                try {
                    const key = await uploadFile(values.profile_picture, 'image');

                    onboardingFormData.append('user_id', result.user_id.toString());
                    onboardingFormData.append('profile_picture', key);

                    await authAPI.onboarding(onboardingFormData);
                }
                catch (error) {
                    console.error('Profile picture upload failed:', error);
                }
            }

            localStorage.clear();
            signupForm.reset();
            navigate('/auth/login');
        }
        catch {
            setServerError('Submission failed. Please try again.');
        }
    }

    return (
        <div id='signup_page'>
            <h1><b>SignUp</b></h1>

            <Form {...signupForm}>
                <form onSubmit={signupForm.handleSubmit(onSubmit)} className='space-y-6 max-w-md'>
                    <FormField
                        control={signupForm.control}
                        name='user_type'
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>User Type</FormLabel>
                                <Select onValueChange={field.onChange} defaultValue={field.value}>
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder='Select user type' />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        <SelectItem value='admin'>Admin</SelectItem>
                                        <SelectItem value='client'>Client</SelectItem>
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={signupForm.control}
                        name='name'
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Name</FormLabel>
                                <FormControl>
                                    <Input placeholder='Enter your name' {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={signupForm.control}
                        name='email'
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Email</FormLabel>
                                <FormControl>
                                    <Input placeholder='Enter email' {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={signupForm.control}
                        name='password'
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Password</FormLabel>
                                <FormControl>
                                    <Input placeholder='Enter password' {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={signupForm.control}
                        name='date_of_birth'
                        render={({ field }) => (
                            <FormItem className='flex flex-col'>
                                <FormLabel>Date of birth</FormLabel>
                                <FormControl>
                                    <DatePicker value={field.value} onChange={field.onChange} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={signupForm.control}
                        name='profile_picture'
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Profile Picture</FormLabel>
                                <FormControl>
                                    <MediaPicker
                                        value={field.value}
                                        onChange={field.onChange}
                                        accept='image/*'
                                        variant='avatar'
                                    />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {serverError && (
                        <p className='text-sm text-destructive'>{serverError}</p>
                    )}

                    <Button type='submit' disabled={signupForm.formState.isSubmitting}>
                        {signupForm.formState.isSubmitting ? 'Submitting...' : 'Submit'}
                    </Button>
                </form>
            </Form>
        </div>
    )
}