from sachin import predict
j=1
while(j==1):
    n=input("Enter Image Name That You Want To Identify = ")
    d=input("Enter Path of Sample Image = ")
    k=predict(d,n)
    print('\n',k)
    j=int(input("\nEnter 1 to continue= "))
