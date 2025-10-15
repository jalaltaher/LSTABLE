

# =============================================================================
# Examples calibration
# =============================================================================

def  integration_bound(Delta,process='default'):
    if process=='default':
        return None 
    elif process=='brownian':
        if Delta==10:
            return 5,1000
        if Delta==1:
            return 10,1000
        if Delta==0.1:
            return 100,1000
        if Delta==0.01:
            return 50,1000 
        
    elif process=='cauchy':
        if Delta==10:
            return 5,1000
        if Delta==1:
            return 10,1000
        if Delta==0.1:
            return 100,1000
        if Delta==0.01:
            return 500,1000
        
    elif process=='levy':
        if Delta==10:
            return 0.01,5000
        if Delta==1:
            return 1,5000#100,1000
        if Delta==0.1:
            return 100,5000
        if Delta==0.01:
            return 5000,10000 #100,1000
        
    elif process=='stable FV':
        if Delta==10:
            return 0.1,5000
        if Delta==1:
            return 1,5000#100,1000
        if Delta==0.1:
            return 50,5000
        if Delta==0.01:
            return 5000,10000 #100,1000
    elif process=='stable IV':
        if Delta==10:
            return 0.5,1000
        if Delta==1:
            return 1,1000#100,1000
        if Delta==0.1:
            return 10,1000
        if Delta==0.01:
            return 5000,10000 #100,1000

def bound_cf(Delta):
    if Delta==0.1:
        return 100,0.01
    elif Delta==1:
        return 10,0.001
    elif Delta==10:
        return 1,0.0001
    elif Delta==5:
    
                                            return 3,0.0002

    
def density_bound(Delta,process='default',P=1):
    if process=='default':
        return None
    elif process=='brownian':
        if Delta==10:
            return -10,10
        if Delta==5:
            return -10,10
        if Delta==1:
            return -3,3
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.5,0.5 #-0.2,.2
        
    elif process=='cauchy':
        if Delta==10:
            return -100,100
        if Delta==5:
            return -20,20
        if Delta==1:
            return -5,5
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.2,0.2
        
    elif process=='levy':
        if Delta==10:
            return 100,10000
        if Delta==5:
            return 0,2000
        if Delta==1:
            return -2*Delta*P,100 #P=1 or mutiply both by P
        if Delta==0.1:
            return -2*Delta*P,1.5
        if Delta==0.01:
            return -2*Delta*P,0.1 
        
    elif process=='stable FV':
        if Delta==10:
            return -2000,2000
        if Delta==5:
            return -500,500
        if Delta==1:
            return -50,50
        if Delta==0.1:
            return -2,2
        if Delta==0.01:
            return -0.25,0.25
    elif process=='stable IV':
        if Delta==10:
            return -100,50
        if Delta==5:
            return -60,60
        if Delta==1:
            return -40,40
        if Delta==0.1:
            return -10,10
        if Delta==0.01:
            return -0.25,0.25
    elif process=='tempered stable FV':
        if Delta==10:
            return -2000,2000
        if Delta==5:
            return -500,500
        if Delta==1:
            return -50,50
        if Delta==0.1:
            return -2,2
        if Delta==0.01:
            return -0.25,0.25
    elif process=='tempered stable Cauchy':
        if Delta==10:
            return -100,100
        if Delta==5:
            return -20,20
        if Delta==1:
            return -5,5
        if Delta==0.1:
            return -1.5,1.5
        if Delta==0.01:
            return -0.2,0.2
    elif process=='tempered stable IV':
        if Delta==10:
            return -100,50
        if Delta==5:
            return -60,60
        if Delta==1:
            return -40,40
        if Delta==0.1:
            return -10,10
        if Delta==0.01:
            return -0.25,0.25
        