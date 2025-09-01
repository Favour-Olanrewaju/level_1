# USSD with OOP

import time

import datetime as dt
current_dt = dt.datetime.now()

class USSD:
    
    def __init__(self):
        self.network = "MTN"
        self.code = "*312#"

        self.main_page()


    def main_page (self):
        print(f"Welcome to {self.network}")
        user = input("code :").strip()
        while user !=self.code:
            print("Invalid code")
            self.main_page()
        self.dash_board()

    def dash_board(self):
        time.sleep(1)
        print("""
        Welcome!
                1. Data plans
                2. Get 1.5GB for N300
                3. Borrow credit/Recharge
                """)
        
        
        user = input('Option: ')
        if user == '1':
            self.data_plan()
        elif user == '2':
            self.bonus()
        elif user == '3':
            self.borrow()
        else:
            print("Invalid code...")
        
        
    def data_plan(self):
        time.sleep(1)
        print("""
        Buy Data Plans
            1. Daily
            2. Weekly
            3. Monthly
            0. back
        """)
        user2 = input('Option: ')
        if user2 == '1':
            self.daily()
        elif user2 == '2':
            self.weekly()
        elif user2 == '3':
            self.monthly()
        elif user2 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
            
            
    def daily(self):
        time.sleep(1)
        print("""
        1. N50 for 40MB
        2. N100 for 100MB
        0. back
        """)
        user3 = input('Option: ')
        if user3 == '1':
            self.N50()
        elif user3 == '2':
            self.N100()
        elif user3 == '0':
                self.dash_board()
        else:
                print("Invalid Input...")    


    def N50(self):
        time.sleep(1)
        print("""
            You will be charged N50 for the purchase of 40MB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0.back
        """)
        user4 = input('option: ')
        if user4 == '1':
            self.auto_renew()
        elif user4 =="2":
            self.one_off()
        elif user4 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
            
    def auto_renew(self):  
        time.sleep(1)
        print(f"""Yello!,Your purchase of N50 for 40MB on {dt.datetime.now()} was successful. Dial *310# to check your DATA balanced..Replay with N03 to stop Auto-Renewal.""")

    def one_off(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N50 for 40MB on {dt.datetime.now()}was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    
    def N100(self):
        time.sleep(1)
        print("""
            You will be charged N100 for the purchase of 100MB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0.back
        """)
        user5 = input('Option: ')
        if user5 == '1':
            self.Auto_renew()
        elif user5 == '2':
            self.One_off()
        elif user5 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def Auto_renew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N100 for 100MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def One_off(self):
        time.sleep(1)
        print(f"""
            Yello!
                Your purchase of N100 for 100MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    def weekly(self):
        time.sleep(1)
        print('''
                1. N350 for 350MB
                2. N500 for 750MB
                3. N1,000 for 1.5GB
                0.back
            ''')
        user6 = input('Option: ')
        if user6 == '1':
            self.N350()
        elif user6 == '2':
            self.N500()
        elif user6 == '3':
            self.N1000()
        elif user6 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    def N350(self):
        time.sleep(1)
        print("""
            You will be charged N350 for the purchase of 350MB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0.back
        """)
        user7 = input('Option: ')
        if user7 == '1':
            self.Auto_Renew()
        elif user7 == '2':
            self.One_Off()
        elif user7 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
            

    def Auto_Renew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N350 for 350MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def One_Off(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N350 for 350MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    
    def N500(self):
        time.sleep(1)
        print("""
            You will be charged N500 for the purchase
            2. One-off
            0.back
        """)
        user8 = input('Option: ')
        if user8 == '1':
            self.AutoRenew()
        elif user8 == '2':
            self.OneOff()
        elif user8 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
            
    
    def AutoRenew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N500 for 750MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def OneOff(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N500 for 750MB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    
    def N1000(self):
        time.sleep(1)
        print("""
            You will be charged N1000 for the purchase of 1.5GB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0. back
        """)
        user9 = input('Option: ')
        if user9 == '1':
            self.autoRenew()
        elif user9 == '2':
            self.oneOff()
        elif user9 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def autoRenew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1000 for 1.5GB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def oneOff(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1000 for 1.5GB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    def monthly(self):
        time.sleep(1)
        print('''
                1. N1,000 for 1.2GB
                2. N1,200 for 1.5GB
                3. N1,500 for 5GB
                0.back
            ''')
        mon = input('Option: ')
        if mon == '1':
            self.n1000()
        elif mon == '2':
            self.n1200()
        elif mon == '3':
            self.n1500()
        elif mon == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def n1000(self):
        time.sleep(1)
        print("""
            You will be charged N1000 for the purchase of 1.2GB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0. back
        """)
        mon1 = input('Option: ')
        if mon1 == '1':
            self.autORenew()
        elif mon1 == '2':
            self.onEOff()
        elif mon1 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def autORenew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1000 for 1.2GB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def onEOff(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1000 for 1.2GB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    
    def n1200(self):
        time.sleep(1)
        print("""
            You will be charged N1200 for the purchase of 1.5GB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0.back
        """)
        mon2 = input('Option: ')
        if mon2 == '1':
            self.autO_Renew()
        elif mon2 == '2':
            self.onE_Off()
        elif mon2 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def autO_Renew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1200 for 1.5GB on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def onE_Off(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1200 for 1.5GB on {dt.datetime.now()}was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
        """)
    
    
    def n1500(self):
        time.sleep(1)
        print("""
            You will be charged N1500 for the purchase of 5GB Daily plan. To proceed, select:
            1. Auto-Renew
            2. One-off
            0.back
        """)
        mon3 = input('Option: ')
        if mon3 == '1':
            self.autO__Renew()
        elif mon3 == '2':
            self.onE__Off()
        elif mon3 == '0':
                self.dash_board()
        else:
            print('Invalid code...')
    
    
    def autO__Renew(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1500 for 5GB on  {dt.datetime.now()}was successful.
                Dial *310# to check your DATA balanced..
                Replay with N03 to stop Auto-Renewal.
        """)
        
    def onE__Off(self):
        time.sleep(1)            
        print(f"""
            Yello!
                Your purchase of N1500 for 5GB on{dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
                Replay with yes3 to enter Auto-Renewal.
            """)
    
    
    def bonus(self):
            time.sleep(1)
            print("""
                Get DOUBLE DATA when you buy 1.5GB at N1,200. You get 1.5GB + 5 voice mins & amp;EXTRA 1.5GB data bonus. Data is valid for 30days.
                    1.Activate
                    0. Back
                """)
            
            bonus= input('Option: ')
            if bonus == '1':
                self.activate()
            elif bonus == '0':
                self.dash_board()
            else:
                print("Invalid Input")


    def activate(self):
        time.sleep(1)
        print(f"""
            Yello!
                Your purchase of N1,200 for 1.5GB + 5 voice mins & amp;EXTRA 1.5GB data bonus on {dt.datetime.now()} was successful.
                Dial *310# to check your DATA balanced..
        """)


    def borrow(self):
        time.sleep(1)
        print('''
                U are not eligible to Borrow Airtime/Data!!
            ''')


ussd = USSD()

