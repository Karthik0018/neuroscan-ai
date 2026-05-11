from supabase import create_client
import os

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

print(f"URL: {SUPABASE_URL}")
print(f"Key starts with: {SUPABASE_KEY[:20]}...")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table('users').select('*', count='exact').execute()
    print(f"✅ Connected successfully! User count: {result.count}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
