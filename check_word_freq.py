import re

text = open('Tagalog_Balita_Texts_Balanced.txt', 'r', encoding='utf-8').read().lower()

pairs = [
    ('bote', 'buti'),
    ('asero', 'asido'),
    ('boto', 'buto')
]

print("Word frequencies in Tagalog_Balita_Texts_Balanced.txt:")
print("=" * 50)
for w1, w2 in pairs:
    count1 = len(re.findall(r'\b' + w1 + r'\b', text))
    count2 = len(re.findall(r'\b' + w2 + r'\b', text))
    total = count1 + count2
    ratio = f"{count1}:{count2}" if count2 > 0 else f"{count1}:0"
    print(f"{w1:8} = {count1:4} | {w2:8} = {count2:4} | Total: {total:4} | Ratio: {ratio}")
