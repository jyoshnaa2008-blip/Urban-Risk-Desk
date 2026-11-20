import sqlite3
conn = sqlite3.connect('scenarios.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', tables)
for table in tables:
    cursor.execute(f'SELECT * FROM {table[0]}')
    rows = cursor.fetchall()
    print(f'{table[0]}: {rows}')
conn.close()