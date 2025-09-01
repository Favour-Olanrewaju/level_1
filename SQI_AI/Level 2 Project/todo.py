import time as t

class ToDo:
    def __init__(self):
        self.landing_page()
    
    def landing_page(self):
        print("To-Do List".center(50,"_"))
        t.sleep(1)
        
        print('''
        1. Add to list
        2. Edit list
        3. Display list
        4. Delete item
        5. Delete list
            
            ''')
        t.sleep(1)
        self.operation()
        
    
    def operation(self):
        User = input('Enter operation: ')
        
        if User == '1':
            self.add_to_list()
        elif User == '2':
            print('You cannot edit an empty list ')
            self.operation()
        elif User == '3':
            print('You cannot display an empty list ')
            self.operation()
        elif User == '4':
            print('No list available ')
            self.operation()
        elif User == '5':
            print('You cannot edit an empty list ')
            self.operation()
            # self.delete_list()
        else:
            print('Invalid Input...')
            
            t.sleep(1)
            
            self.landing_page()
    
    
    def add_to_list(self):
        self.DataBase = DataBase = []
        User = int(input('How many items do you want to add to the list? (U can only add up to five items): '))
        for _ in range(User):
            add = input('Enter items you want to add: ').strip().title()
            DataBase.append(add)

        x = input("Do u want to load your list before saving it(Y/N):").strip().lower()
        if x =="y":
            print(DataBase)
        elif x =='n':
            print('saved!')
        else:
            print("Invalid Input..")
            t.sleep(1)
            print("Items not Saved!")
            self.add_to_list()
        
        while True:
            y = input("Do you want to perform another action(y/n): ").strip().lower()
            if y == "y":
                print('''
                    1. Edit list
                    2. Display list
                    3. Delete item
                    4. Delete list
                    5. Exit
                ''')
            elif y == 'n':
                print(f'Summary of your To Do list {self.DataBase}')
                exit()
            else:
                print('Invalid Input!')
            Action = input('Enter your action: ')
            if Action== '1':
                self.edit_list()
            elif Action == '2':
                self.display_list()
            elif Action == '3':
                self.delete_item()
            elif Action == '4':
                self.delete_list()
            elif Action == '5':
                exit()
            else:
                t.sleep(1)
                print('Invalid Input')
                
    
    def edit_list(self):
        print(self.DataBase)
        Edit = input("Enter the index number of the list items u want to edit: ")
        for num in Edit:
                if num =="0":
                    print(f' editing {self.DataBase[0]}..')
                    edit2 = input("type what u want to edit: ").strip().lower().title()
                    self.DataBase[0]= edit2
                    print(self.DataBase)
                elif Edit == "1":
                    print(f' editing {self.DataBase[1]}..')
                    # print(self.DataBase[1])
                    edit2 = input("type what u want to edit: ").strip().lower().title()
                    self.DataBase[1]= edit2
                    print(self.DataBase)
                elif Edit == "2":
                    print(f' editing {self.DataBase[2]}..')
                    # print(self.DataBase[1])
                    edit2 = input("type what u want to edit: ").strip().lower().title()
                    self.DataBase[2]= edit2
                    print(self.DataBase)
                elif Edit == "3":
                    print(f' editing {self.DataBase[3]}..')
                    # print(self.DataBase[1])
                    edit2 = input("type what u want to edit: ").strip().lower().title()
                    self.DataBase[3]= edit2
                    print(self.DataBase)
                elif Edit == "4":
                    print(f' editing {self.DataBase[4]}..')
                    # print(self.DataBase[1])
                    edit2 = input("type what u want to edit: ").strip().lower().title()
                    self.DataBase[4]= edit2
                    print(self.DataBase)
                else :
                    print('Wrong Input')
                    t.sleep(1)
                    self.edit_list()
                    
                    
    def display_list(self):
        print(self.DataBase)
        

    def delete_item(self):
        print(self.DataBase)
        Delete = input("Enter the index number of the list item u want to delete: ")
        if Delete == "0":
            self.DataBase.pop(0)
            print(self.DataBase)
        elif Delete == "1":
            self.DataBase.pop(1)
            print(self.DataBase)
        elif Delete == "2":
            self.DataBase.pop(2)
            print(self.DataBase)
        elif Delete == "3":
            self.DataBase.pop(3)
            print(self.DataBase)
        elif Delete == "4":
            self.DataBase.pop(4)
            print(self.DataBase)
        else:
            t.sleep(1)
            print('Wrong Input')
            self.delete_item()
            

    def delete_list(self): 
        clear = input("Do u really want to clear your To-Do List(y/n): ").strip().lower()
        if clear == "y":
            self.DataBase.clear()
            print(self.DataBase)
            print("To-Do List has been deleted...")
        elif clear == "n":
            print("cancelled")
        else:
            print('Invalid Input')
            self.delete_list()
            
            
toDo = ToDo()