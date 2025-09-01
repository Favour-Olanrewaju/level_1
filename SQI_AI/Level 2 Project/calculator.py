# This is a calculator project

import math
import time as t
class calculator:
    def __init__(self):
        print('Calculator'.center(50,'_'))
        t.sleep(1)
        self.landingPage()
        
    def landingPage(self):
        print('''
            1. Basic Calculator
            2. Advanced Calculator
            3. Exit
            ''')
        user = input('Prompt: ')
        if user == "1":
            self.Basic()
        elif user == '2':
            self.Advanced()
        elif user == '3':
            exit()
        else:
            print('Wrong Input')
            t.sleep(1)
            self.landingPage()
        
    def Basic(self):
        print('''
        1. Addition
        2. Subtraction
        3. Division
        4. Multiplication
        5. Exponential 
        6. Floor division
        7. Square root
        8.exit
        
        ''')
        t.sleep(1)
        while True:
            user =input('Prompt: ').strip()
            if user == '1':
                self.addition()
            elif user == '2':
                self.subtraction()
            elif user == '3':
                self.division()
            elif user == '4':
                self.multiplication()
            elif user == '5':
                self.exponential()
            elif user == '6':
                self.Floor_division()
            elif user == '7':
                self.square_root()
            elif user == '8':
                # print("Exiting the calculator.")
                t.sleep(1)
                # print('...')
                # exit()
                self.landingPage()
            else:
                print('Invalid input...')
                t.sleep(1)
            
    def addition(self):
        val1 =eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 + val2 
        t.sleep(1)
        print(f'{val1} + {val2} = {result}')
    
    def subtraction(self):
        val1 = eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 - val2 
        t.sleep(1)
        print(f'{val1} - {val2} = {result}')
    
    def division(self):
        val1 = eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 / val2 
        t.sleep(1)
        print(f'{val1} / {val2} = {result}')
        
    def multiplication(self):
        val1 = eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 * val2 
        t.sleep(1)
        print(f'{val1} * {val2} = {result}')   
        
    def exponential(self):
        val1 = eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 ** val2 
        t.sleep(1)
        print(f'{val1} ** {val2} = {result}')
        
    def floor_division(self):
        val1 = eval(input('value1: ').strip())
        val2 = eval(input('value2: ').strip())
        result = val1 // val2 
        t.sleep(1)
        print(f'{val1} / {val2} = {result}')
        
    def square_root(self):
        val1 = eval(input('value1: ').strip())
        # val2 = eval(input('value2: ').strip())
        result = (val1)**(1/2)
        t.sleep(1)
        print(f'Square root of {val1} = {result}')    
        
        
    def Advanced(self):
        print("Welcome to the Advanced Calculator!")
        t.sleep(1)
        print('This advanced calculator can do all arithmetics operation, including calculating complex numbers.')
        t.sleep(1)
        print('It does not work for trigonometric and log functions')
        t.sleep(1)

        while True:
            expression = input("Enter a mathematical expression (or 'exit' to quit): ")

            if expression.lower() == 'exit':
                # print("Exiting the calculator")
                # print('...')
                exit()
                

            try:
                result = eval(expression)
                # print("Result:", result)
                print(f'the result is: {result}')
            except Exception as e:
                print("Error:", e)
        
        
        
if __name__ == "__main__":
    calc = calculator()
        
        
        
        
    