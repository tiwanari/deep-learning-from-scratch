# -*- coding: utf-8 -*-

class MulLayer(object):
    def __init__(self, ):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x*y

    def backward(self, dout):
        # z = xy
        # d: partial differential
        # dz/dy = x
        # dz/dx = y
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer(object):
    def __init__(self, ):
        pass
    def forward(self, x, y):
        return x + y
    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy


if __name__ == '__main__':
    # buy apple
    apple = 100
    num_apple = 2
    tax = 1.1
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, num_apple)
    price = mul_tax_layer.forward(apple_price, tax)
    print (price)

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, d_num_apple = mul_apple_layer.backward(dapple_price)

    print(dapple, d_num_apple, dtax)

    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    # layer
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()
    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num) #(1)
    orange_price = mul_orange_layer.forward(orange, orange_num) #(2)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price) #(3)
    price = mul_tax_layer.forward(all_price, tax) #(4)
    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice) #(4)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price) #(2)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price) #(1)
    print(price) # 715
    print(dapple_num, dapple, dorange, dorange_num, dtax)

