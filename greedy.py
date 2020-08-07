# This code gives total hit of 502120 for 1st criterias 


import pandas as pd
class Greedy:
    def __init__(self,dataset,revenuePercent,quantityPercent):
        self.dataset = dataset
        self.revenuePercent = revenuePercent
        self.quantityPercent = quantityPercent
    
    def getNewDf(self):
        itemId = self.dataset['Item_id']
        data = pd.DataFrame(itemId , columns= ['Item_id'])
        data['Revenue 1'] = None # stores extra revenue generated by choosing P1 instead of Base Revenue
        data['Quantity 1'] = None # stores extra quantity generated by choosing P1 instead of Base quatity
        data['hit 1'] = None # hit if we choose Price 1
        data['RH1'] = None # Extra Revenue by hit ratio
        data['QH1'] = None # Extra Quantity by hit ratio
        data['Revenue 2'] = None
        data['Quantity 2'] = None
        data['hit 2'] = None
        data['RH2'] = None
        data['QH2'] = None
        data['Revenue 3'] = None
        data['Quantity 3'] = None
        data['hit 3'] = None
        data['RH3'] = None
        data['QH3'] = None
        data['Revenue 4'] = None
        data['Quantity 4'] = None
        data['hit 4'] = None
        data['RH4'] = None
        data['QH4'] = None
        # Computation of all the parameters
        for i in range(len(data['Item_id'])):
            data['Revenue 1'][i] = (self.dataset['Price1'][i]*self.dataset['Units1'][i]) - (self.dataset['Base_Price'][i]*self.dataset['Base_Units'][i])
            data['Quantity 1'][i] = self.dataset['Units1'][i]-self.dataset['Base_Units'][i]
            data['hit 1'][i] = (self.dataset['Base_Price'][i] - self.dataset['Price1'][i])*self.dataset['Units1'][i]
            data['RH1'][i] = (data['Revenue 1'][i]/data['hit 1'][i])
            data['QH1'][i] = data['Quantity 1'][i]/data['hit 1'][i]
            data['Revenue 2'][i] = (self.dataset['Price2'][i]*self.dataset['Units2'][i]) - (self.dataset['Base_Price'][i]*self.dataset['Base_Units'][i])
            data['Quantity 2'][i] = self.dataset['Units2'][i]-self.dataset['Base_Units'][i]
            data['hit 2'][i] = (self.dataset['Base_Price'][i] - self.dataset['Price2'][i])*self.dataset['Units2'][i]
            data['RH2'][i] = (data['Revenue 2'][i]/(data['hit 2'][i]))
            data['QH2'][i] = data['Quantity 2'][i]/(data['hit 2'][i])
            data['Revenue 3'][i] = (self.dataset['Price3'][i]*self.dataset['Units3'][i]) - (self.dataset['Base_Price'][i]*self.dataset['Base_Units'][i])
            data['Quantity 3'][i] = self.dataset['Units3'][i]-self.dataset['Base_Units'][i]
            data['hit 3'][i] = (self.dataset['Base_Price'][i] - self.dataset['Price3'][i])*self.dataset['Units3'][i]
            data['RH3'][i] = data['Revenue 3'][i]/(data['hit 3'][i])
            data['QH3'][i] = data['Quantity 3'][i]/(data['hit 3'][i])
            data['Revenue 4'][i] = (self.dataset['Price4'][i]*self.dataset['Units4'][i]) - (self.dataset['Base_Price'][i]*self.dataset['Base_Units'][i])
            data['Quantity 4'][i] = self.dataset['Units4'][i]-self.dataset['Base_Units'][i]
            data['hit 4'][i] = (self.dataset['Base_Price'][i] - self.dataset['Price4'][i])*self.dataset['Units4'][i]
            data['RH4'][i] = data['Revenue 4'][i]/(data['hit 4'][i])
            data['QH4'][i] = data['Quantity 4'][i]/(data['hit 4'][i])

        #print(data)
        #data.to_csv("data.csv")
        # Generating seperate tables for RH and QH ratio for easier access later
        data2 = pd.DataFrame(columns = ['Item_id','RH_ratio','Price ID'])
        data3 = pd.DataFrame(columns = ['Item_id','QH_ratio','Price ID'])
        for i in range(len(data['RH1'])):
            data2 = data2.append({'Item_id':i+1,'RH_ratio':data['RH1'][i],'Price ID':1},ignore_index=True)
            data2 = data2.append({'Item_id':i+1,'RH_ratio':data['RH2'][i],'Price ID':2},ignore_index=True)
            data2 = data2.append({'Item_id':i+1,'RH_ratio':data['RH3'][i],'Price ID':3},ignore_index=True)
            data2 = data2.append({'Item_id':i+1,'RH_ratio':data['RH4'][i],'Price ID':4},ignore_index=True)
            data3 = data3.append({'Item_id':i+1,'QH_ratio':data['QH1'][i],'Price ID':1},ignore_index=True)
            data3 = data3.append({'Item_id':i+1,'QH_ratio':data['QH2'][i],'Price ID':2},ignore_index=True)
            data3 = data3.append({'Item_id':i+1,'QH_ratio':data['QH3'][i],'Price ID':3},ignore_index=True)
            data3 = data3.append({'Item_id':i+1,'QH_ratio':data['QH4'][i],'Price ID':4},ignore_index=True)
        #print(data2)
        #print("NEW DF GENERATED SUCCESSFULLY")
        return data,data2,data3

    def getRequiredRevenueQuantity(self):
        # Computes the extra revenue & quantity we want 
        # i.e x% of total base revenue & y% of total base quantity
        totalRevenue=0
        totalQuantity=0
        for i in range(len(self.dataset['Item_id'])):
            totalRevenue+=self.dataset['Base_Price'][i]*self.dataset['Base_Units'][i]
            totalQuantity+=self.dataset['Base_Units'][i]
        totalRevenue = int(totalRevenue*(self.revenuePercent/100))
        totalQuantity = int(totalQuantity*(self.quantityPercent/100))
        totalRevenue+=1
        totalQuantity+=1
        
        return totalRevenue,totalQuantity
    
    def initialComputation(self,requiredQuantity,requiredRevenue,revenueDf,totalhit,itemsDone):
        # Just an extra check for optimisation , if worst hit i.e hit if we choose product 4 is less than 1000 , it can be choosen
        worsthit = [0 for i in range(len(self.dataset['Item_id']))]
        for i in range(len(self.dataset['Item_id'])):
            worsthit[i] = abs((self.dataset['Price4'][i]-self.dataset['Base_Price'][i])*self.dataset['Units4'][i])
            if(worsthit[i]<1000):
                #print(revenueDf['Revenue 4'][i])
                
                requiredRevenue-=(revenueDf['Revenue 4'][i])
                requiredQuantity-=(revenueDf['Quantity 4'][i])
                totalhit+=worsthit[i]
                itemsDone[i+1]=4

        return requiredQuantity,requiredRevenue,totalhit,itemsDone
    
    def GreedyApproach(self):

        revenueDf , rhRatioDf , qhRatioDf = self.getNewDf()
        rhRatioDfSorted = rhRatioDf.sort_values('RH_ratio',ascending=False)
        rhRatioDfSorted = rhRatioDfSorted.reset_index(drop=True)
        qhRatioDfSorted = qhRatioDf.sort_values('QH_ratio',ascending=False)
        qhRatioDfSorted = qhRatioDfSorted.reset_index(drop=True)
        #rhRatioDfSorted.to_csv("abc.csv")
        requiredRevenue , requiredQuantity = self.getRequiredRevenueQuantity()
        #print("REQUIRED REVENUE = ",requiredRevenue)
        #print("REQUIRED QUANTITY = ",requiredQuantity)
        #print(rhRatioDfSorted)
        row = 0
        itemsDone = dict()
        totalHit = 0
        requiredQuantity , requiredRevenue ,totalHit ,itemsDone= self.initialComputation(requiredQuantity,requiredRevenue,revenueDf , totalHit,itemsDone)
        #print("REQUIRED QUANTITY",requiredQuantity)
        #print("REQUIRED REVENUE",requiredRevenue)
        #print("Total hit",totalHit)


        # First finish off the quantity criteria
        
        while(requiredQuantity>0):
            item = qhRatioDfSorted['Item_id'][row]
            priceId = qhRatioDfSorted['Price ID'][row]
            #print("ITEM =",item)
            #print("PRICE ID=",priceId)
            if((item in itemsDone)):

                # If item was already encountered earlier then we need to subtract previous instance and add new instance
                # If item 1 P1 was viable earlier but now algorithm predicts P2 is also viable then subtract P1 instance and add P2 instance
                
                
                if(itemsDone[item]!=priceId):
                    donePriceId = itemsDone[item]
                    if(priceId==2 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 2'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 2'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 2'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=2
                    
                    elif(priceId==3 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 3'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 3'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 3'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=3
                    elif(priceId==4 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=4
                    elif(priceId==3 and donePriceId==2):
                        requiredRevenue-=(revenueDf['Revenue 3'][item-1]-revenueDf['Revenue 2'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 3'][item-1]-revenueDf['Quantity 2'][item-1])
                        totalHit+=(revenueDf['hit 3'][item-1]-revenueDf['hit 2'][item-1])
                        itemsDone[item]=3
                    elif(priceId==4 and donePriceId==2):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 2'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 2'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 2'][item-1])
                        itemsDone[item]=4
                    elif(priceId==4 and donePriceId==3):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 3'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 3'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 3'][item-1])
                        itemsDone[item]=4
                    
                
            else:
                if(priceId==1):
                    requiredRevenue-=revenueDf['Revenue 1'][item-1]
                    requiredQuantity-=revenueDf['Quantity 1'][item-1]
                    totalHit+=revenueDf['hit 1'][item-1]
                    itemsDone[item]=1
                elif(priceId==2):
                    requiredRevenue-=revenueDf['Revenue 2'][item-1]
                    requiredQuantity-=revenueDf['Quantity 2'][item-1]
                    totalHit+=revenueDf['hit 2'][item-1]
                    itemsDone[item]=2
                elif(priceId==3):
                    requiredRevenue-=revenueDf['Revenue 3'][item-1]
                    requiredQuantity-=revenueDf['Quantity 3'][item-1]
                    totalHit+=revenueDf['hit 3'][item-1]
                    itemsDone[item]=3
                elif(priceId==4):
                    requiredRevenue-=revenueDf['Revenue 4'][item-1]
                    requiredQuantity-=revenueDf['Quantity 4'][item-1]
                    totalHit+=revenueDf['hit 4'][item-1]
                    itemsDone[item]=4
            row+=1



        # checking for the revenue criteria , although it should have been met after we exit quantity while loop
        row=0
        while(requiredRevenue>0):
            item = rhRatioDfSorted['Item_id'][row]
            priceId = rhRatioDfSorted['Price ID'][row]
            #print("ITEM =",item)
            #print("PRICE ID=",priceId)
            if((item in itemsDone)):
                
                
                if(itemsDone[item]!=priceId):
                    donePriceId = itemsDone[item]

                    if(priceId==2 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 2'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 2'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 2'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=2
                    elif(priceId==donePriceId):
                        continue
                    
                    elif(priceId==3 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 3'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 3'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 3'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=3
                    elif(priceId==4 and donePriceId==1):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 1'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 1'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 1'][item-1])
                        itemsDone[item]=4
                    elif(priceId==3 and donePriceId==2):
                        requiredRevenue-=(revenueDf['Revenue 3'][item-1]-revenueDf['Revenue 2'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 3'][item-1]-revenueDf['Quantity 2'][item-1])
                        totalHit+=(revenueDf['hit 3'][item-1]-revenueDf['hit 2'][item-1])
                        itemsDone[item]=3
                    elif(priceId==4 and donePriceId==2):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 2'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 2'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 2'][item-1])
                        itemsDone[item]=4
                    elif(priceId==4 and donePriceId==3):
                        requiredRevenue-=(revenueDf['Revenue 4'][item-1]-revenueDf['Revenue 3'][item-1])
                        requiredQuantity-=(revenueDf['Quantity 4'][item-1]-revenueDf['Quantity 3'][item-1])
                        totalHit+=(revenueDf['hit 4'][item-1]-revenueDf['hit 3'][item-1])
                        itemsDone[item]=4
                    
                
            else:
                if(priceId==1):
                    requiredRevenue-=revenueDf['Revenue 1'][item-1]
                    requiredQuantity-=revenueDf['Quantity 1'][item-1]
                    totalHit+=revenueDf['hit 1'][item-1]
                    itemsDone[item]=1
                elif(priceId==2):
                    requiredRevenue-=revenueDf['Revenue 2'][item-1]
                    requiredQuantity-=revenueDf['Quantity 2'][item-1]
                    totalHit+=revenueDf['hit 2'][item-1]
                    itemsDone[item]=2
                elif(priceId==3):
                    requiredRevenue-=revenueDf['Revenue 3'][item-1]
                    requiredQuantity-=revenueDf['Quantity 3'][item-1]
                    totalHit+=revenueDf['hit 3'][item-1]
                    itemsDone[item]=3
                elif(priceId==4):
                    requiredRevenue-=revenueDf['Revenue 4'][item-1]
                    requiredQuantity-=revenueDf['Quantity 4'][item-1]
                    totalHit+=revenueDf['hit 4'][item-1]
                    itemsDone[item]=4
            row+=1

            

        return itemsDone,totalHit,requiredRevenue,requiredQuantity


    def getSolution(self):


        # Dumping results in CSV File here
        itemsDone,totalHit,requiredRevenue,requiredQuantity = self.GreedyApproach()
        print("Total Hit = ",totalHit)
        print("Extra Revenue Generated",abs(requiredRevenue))
        print("Extra Required Quantity",abs(requiredQuantity))
        itemId = self.dataset['Item_id']
        solution = pd.DataFrame(itemId , columns= ['Item_id'])
        solution['Price'] = "Base_Price"
        print(type(itemsDone))
        for key,value in itemsDone.items():
            key = int(key)
            value = int(value)
            if(value==1):
                solution['Price'][key-1]="Price1"
            elif(value==2):
                solution['Price'][key-1]="Price2"
            elif(value==3):
                solution['Price'][key-1]="Price3"
            elif(value==4):
                solution['Price'][key-1]="Price4"
        return solution
#####MAIN CODE


# Call the class and pass reqd variables
dataset = pd.read_csv("dataset.csv")
revenuePercent = 10
quantityPercent = 25
greedyClass = Greedy(dataset,revenuePercent,quantityPercent)
submission = greedyClass.getSolution()
submission.to_csv("result.csv")


        


