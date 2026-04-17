import 'dotenv/config';
import { defineConfig } from 'drizzle-kit';

// Helper function to clean the DATABASE_URL
function getDatabaseUrl(): string {
  const url = process.env.DATABASE_URL || '';
  // Remove 'psql ' prefix and quotes if present
  return url
    .replace(/^psql\s*['"]?/i, '')
    .replace(/['"]\s*$/, '')
    .trim();
}

export default defineConfig({
  schema: './config/schema.tsx',
  dialect: 'postgresql',
  dbCredentials: {
    url: getDatabaseUrl(),
  },
});
