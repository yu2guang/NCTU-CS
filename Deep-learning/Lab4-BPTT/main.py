import model

if __name__ == '__main__':

    # parameters
    EPOCH = 15
    ITERATIONS = 1000
    ALPHA = 0.01

    print('---Training---')
    net = model.train(EPOCH, ITERATIONS, ALPHA)

    print('\n---Testing 1(input 1000 to break)---')
    while(1):
        num1 = int(input('\ninput 1st integer:'))
        if(num1==1000): break
        num2 = int(input('input 2nd integer:'))
        ans = model.test(net, num1, num2)
        print('#####Result: %d + %d = %d#####'%(num1, num2, ans))

    print('\n---Testing 2---')
    right = 0
    total = 0
    for num1 in range(0, 128):
        for num2 in range(0, 128):
            ans = model.test(net, num1, num2)
            print('#####Result: %d + %d = %d#####'%(num1, num2, ans))
            if(ans==(num1+num2)):
                right+=1
            total+=1
    print('Accuracy:',right/total)













