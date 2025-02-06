import re
import os
lst = []
for x in os.listdir('D:\\Dataset\\archive\\RAM_GroundedSAM2\\outputs\\segment'):
    # Regular expression to extract the number
    match = re.search(r'_val_(\d+)_', x)

    if match:
        # Extract the matched number
        number = match.group(1)
        lst.append(int(number))  # Output: 00004965
set_lst= set(lst)
for x in range(1,30000):
    if x in set_lst:
        pass
    else:
        print(x,end=',')