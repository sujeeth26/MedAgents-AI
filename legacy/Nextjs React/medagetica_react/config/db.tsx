import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';

// Helper function to clean the DATABASE_URL
function getDatabaseUrl(): string {
  const url = process.env.DATABASE_URL || '';
  return url
    .replace(/^psql\s*['"]?/i, '')
    .replace(/['"]\s*$/, '')
    .trim();
}

const sql = neon(getDatabaseUrl());
export const db = drizzle({ client: sql });
