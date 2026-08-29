import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
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

const loginFormSchema = z.object({
    user_type: z.enum(['admin', 'client'], {
        required_error: 'Please select user type'
    }),
    email: z.string().email(),
    password: z.string().min(1, 'Please enter password')
});

type LoginFormValues = z.infer<typeof loginFormSchema>;

export function Login() {
    const { login } = useAuth();
    const navigate = useNavigate();
    const [serverError, setServerError] = useState<string | null>(null);

    const loginForm = useForm<LoginFormValues>({
        resolver: zodResolver(loginFormSchema),
        defaultValues: { email: '', password: '' },
    })

    async function onSubmit(values: LoginFormValues) {
        setServerError(null);

        const formData = new FormData();
        formData.append('user_type', values.user_type)
        formData.append('email', values.email)
        formData.append('password', values.password);

        try {
            const result = await authAPI.login(formData);

            login(
                {
                    user_type: values.user_type,
                    user_id: result.user_id,
                    name: result.name,
                    email: values.email,
                    profile_picture: result.profile_picture
                },
                result.token
            );

            loginForm.reset();
            navigate('/home');
        }
        catch {
            setServerError('Submission failed. Please try again.');
        }
    }

    return (
        <div id='login_page'>
            <h1><b>Login</b></h1>

            <Form {...loginForm}>
                <form onSubmit={loginForm.handleSubmit(onSubmit)} className='space-y-6 max-w-md'>
                <FormField
                        control={loginForm.control}
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
                        control={loginForm.control}
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
                        control={loginForm.control}
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

                    {serverError && (
                        <p className='text-sm text-destructive'>{serverError}</p>
                    )}

                    <Button type='submit' disabled={loginForm.formState.isSubmitting}>
                        {loginForm.formState.isSubmitting ? 'Submitting...' : 'Submit'}
                    </Button>
                </form>
            </Form>
        </div>
    )
}