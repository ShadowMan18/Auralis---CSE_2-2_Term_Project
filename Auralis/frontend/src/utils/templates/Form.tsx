import { useState } from 'react';
import api from '@/services/ApiService.ts';
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

// naming changes: 
// formSchema (e.g. loginFormSchema)
// formValues (e.g. loginFormValues)
// form       (e.g. loginForm)
// formData   (e.g. loginFormData)

// define the form schema
const formSchema = z.object({
    user_type: z.enum(['admin', 'client'], {
        required_error: 'Please select user type'
    }),
    name: z.string()
        .min(1, 'Please enter name')
        .max(30, 'Name is too long'),
    date_of_birth: z.date(),
    profile_picture: z.instanceof(File).optional()
});

type formValues = z.infer<typeof formSchema>;

export function DemoForm() {
    const [serverError, setServerError] = useState<string | null>(null);

    const form = useForm<formValues>({
        resolver: zodResolver(formSchema),
        defaultValues: { name: '' },        // define the defualt values
    })

    async function onSubmit(values: formValues) {
        setServerError(null);

        // add items to formData
        const formData = new FormData();
        formData.append('user_type', values.user_type);
        formData.append('name', values.name);
        formData.append('date_of_birth', format(values.date_of_birth, 'yyyy-MM-dd'));
        if (values.profile_picture) formData.append('profile_picture', values.profile_picture);

        // call appropriate api to submit
        try {
            await api.request('/api/submit', {
                method: 'POST',
                body: formData
            });

            form.reset();
        }
        catch {
            setServerError('Submission failed. Please try again.');
        }
    }

    return (
        <div id='signup_page'>
            <h1><b>SignUp</b></h1>

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className='space-y-6 max-w-md'>
                    {/* selection box */}
                    <FormField
                        control={form.control}
                        name='user_type'    // set the field name (must match with schema)
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>User Type</FormLabel>    {/* set the field label */}
                                <Select onValueChange={field.onChange} defaultValue={field.value}>
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder='Select user type' />  {/* set the selection prompt */}
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        <SelectItem value='admin'>Admin</SelectItem>        {/* set the selection items */}
                                        <SelectItem value='client'>Client</SelectItem>
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {/* text field */}
                    <FormField
                        control={form.control}
                        name='name'     // set the field name (must match with schema)
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Name</FormLabel>      {/* set the field label */}
                                <FormControl>
                                    <Input placeholder='Enter your name' {...field} />      {/* set the placeholder */}
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {/* date picker */}
                    <FormField
                        control={form.control}
                        name='date_of_birth'    // set the field name (must match with schema)
                        render={({ field }) => (
                            <FormItem className='flex flex-col'>    
                                <FormLabel>Date of birth</FormLabel>    {/* set the field label */}
                                <FormControl>
                                    <DatePicker value={field.value} onChange={field.onChange} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {/* media picker */}
                    <FormField
                        control={form.control}
                        name='profile_picture'  // set the field name (must match with schema)
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Profile Picture</FormLabel>  {/* set the field label */}
                                <FormControl>
                                    <MediaPicker
                                        value={field.value}
                                        onChange={field.onChange}
                                        accept='image/*'    // set the field label
                                        variant='avatar'    // set preview type [avatar: circle, default: rectangle]
                                    />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {serverError && (
                        <p className='text-sm text-destructive'>{serverError}</p>
                    )}

                    <Button type='submit' disabled={form.formState.isSubmitting}>
                        {form.formState.isSubmitting ? 'Submitting...' : 'Submit'}  {/* set the button label */}
                    </Button>
                </form>
            </Form>
        </div>
    )
}