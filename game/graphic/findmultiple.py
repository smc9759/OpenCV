a3=[]
b=1
while b<1000:
    if b%3 ==0:
        a3.append(b)
    b+=1


c5 =[]
b=1
while b<1000:
    if b%5 ==0:
        c5.append(b)
    b+=1

d15 =[]
b=1
while b<1000:
    if (b%5)& (b%3) ==0:
        d15.append(b)
    b+=1

result = 0
for n in range(1,1000):
    if n%3 ==0 or n%5 ==0:
        result += n
print(result)
