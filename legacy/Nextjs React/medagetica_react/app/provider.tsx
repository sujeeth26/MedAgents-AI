// this being used for client side authentication
"use client";
import React, { useState, useEffect } from 'react'
import axios from 'axios';
import { useUser } from '@clerk/nextjs';
import { UserDetailsContext } from '@/context/UserDetailsContext';

export default function Provider({
    children,
  }: Readonly<{
    children: React.ReactNode;
  }>) 
  {
    const { user } = useUser();
    const [userDetail, setUserDetail] = useState<any>();

    useEffect(() => {
        if (user) {
            const CreateNewUser = async () => {
                try {
                    const result = await axios.post('/api/users');
                    console.log(result.data);
                    setUserDetail(result.data);
                } catch (error) {
                    console.error('Error creating/fetching user:', error);
                }
            }
            CreateNewUser();
        }
    }, [user]);

    return (
        <div>
            <UserDetailsContext.Provider value={{ userDetail, setUserDetail }}>
                {children}
            </UserDetailsContext.Provider>
        </div>
    )
}
