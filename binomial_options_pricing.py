import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class BinomialPricing():

    def __init__(self, use_default = False):
        # Flags
        self.PUT = object()
        self.CALL = object()
        self.STOCK = object()
        self.FUTURE = object()
        self.OPTION = object()
        self.CHOOSER = object() # Chooser underlying
        self.EXERCISE = object()
        self.DEBUG = object()

        # Rates
        self.SHORT_RATE = object()
        self.ZCB = object()
        self.SWAP = object()

        # Initialisation
        self.lattices = {}
        
        if use_default:
            # Lattice Paramters (Input)
            self.S0 = 100
            self.R0 = 6 / 100
            self.T = 0.25
            self.sigma = 30 / 100
            self.n = 3 # Maturity of option
            self.n_future = 5 # Maturity of underlying
            self.r = 2 / 100
            self.c = 1 / 100

            # Rates Parameters
            self.N = 1 # Notional / Real Principle
            self.rK = 5 / 100 # Fixed rate
            
            # Option Parameters (Input)
            self.K = 0
            self.option_type = self.CALL
            self.compute_on = self.SWAP
            self.is_american = False

    def override_computed_parameters(self):
        self.u = 1.25
        self.d = 0.9
        self.q = 0.5

    # Must be called if any numerical paramters are changed in the model
    def compute_option_parameters(self, calc_option=False):
        # Computed Values from BSM
        if calc_option:
            calibrated_T = self.T * (self.n/self.n_future)
            self.dt = calibrated_T / self.n
        else:
            self.dt = self.T / self.n_future
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp((self.r - self.c)*self.dt) - self.d) / (self.u - self.d)
        self.R = np.exp(self.r*self.dt) # Risk free compounded periodically
    
    def init_stock_lattice(self):
        self.compute_option_parameters()
        
        size = self.n_future + 1
        # U
        u_values = np.full((size,size),self.u)
        arr = [i for i in reversed(range(size))]
        lower_r =  np.tril(np.tile(arr, size).reshape(size, size).T)
        u_powers = np.flip(lower_r, axis=1)

        # D
        d_values = np.full((size,size),self.d)
        row, col = np.indices((size,size))
        unmasked = col - row
        mask = unmasked < 0
        unmasked[mask] = 0
        d_powers =  np.flip(unmasked.T, axis=1)

        
        lattice = np.power(u_values, u_powers) * np.power(d_values,d_powers) * self.S0
        lattice[(d_powers + u_powers) == 0] = 0
        lattice[self.n_future,0] = self.S0

        self.lattices[self.STOCK] = lattice

    def init_short_rate_lattice(self):
        self.compute_option_parameters()
        self.override_computed_parameters()
        
        size = self.n_future + 1
        # U
        u_values = np.full((size,size),self.u)
        arr = [i for i in reversed(range(size))]
        lower_r =  np.tril(np.tile(arr, size).reshape(size, size).T)
        u_powers = np.flip(lower_r, axis=1)

        # D
        d_values = np.full((size,size),self.d)
        row, col = np.indices((size,size))
        unmasked = col - row
        mask = unmasked < 0
        unmasked[mask] = 0
        d_powers =  np.flip(unmasked.T, axis=1)
        
        lattice = np.power(u_values, u_powers) * np.power(d_values,d_powers) * self.R0
        lattice[(d_powers + u_powers) == 0] = 0
        lattice[self.n_future,0] = self.R0

        self.lattices[self.SHORT_RATE] = lattice

    def init_futures_lattice(self):
        self.compute_option_parameters()
        
        size = self.n_future + 1
        lattice = np.zeros((size,size))
        stock_lattice = self.lattices[self.STOCK]
        lattice[:,-1] = stock_lattice[:,size-1]  # F_T = S_T

        last_row_idx = lattice.shape[0] - 1
        
        for i in reversed(range(1, self.n_future+1)):
            col = lattice[:,i]

            for j in range(i):
                down_s = col[last_row_idx-j]
                up_s = col[last_row_idx-(j+1)]
                lattice[last_row_idx-j,i-1] = self.q * up_s + (1-self.q) * down_s

        self.lattices[self.FUTURE] = lattice

    def init_options_lattice(self, use_short_rate=False):
        self.compute_option_parameters(calc_option=True)
        self.override_computed_parameters()
        
        size = self.n + 1
        lattice = np.zeros((size,size))
        ex_early_lattice = np.zeros((size,size))
        debug = np.zeros((size,size))
  
        stock_lattice = self.lattices[self.compute_on]
        stock_last_row_idx = stock_lattice.shape[0] - 1
        short_rates = self.lattices[self.SHORT_RATE]
        
        # Gets the payouts at expiration period n (+ 1 for zero index)
        if self.option_type is self.CALL:
            lattice[:,-1] = np.maximum(stock_lattice[stock_last_row_idx-size+1:,size-1] - self.K, 0)
        else:
            lattice[:,-1] = np.maximum(self.K - stock_lattice[stock_last_row_idx-size+1:,size-1], 0)  

        last_row_idx = lattice.shape[0] - 1
        short_rates_last_row_idx = short_rates.shape[0] - 1 
        
        for i in reversed(range(0, self.n+1)): 
            col = lattice[:,i]

            for j in range(i):
                down_s = col[last_row_idx-j] 
                up_s = col[last_row_idx-(j+1)] 

                rate = self.R
                
                if use_short_rate:
                    rate = 1+short_rates[short_rates_last_row_idx-j,i-1]
                
                if self.is_american:
                    if self.option_type is self.CALL:
                        ex = stock_lattice[stock_last_row_idx-j,i-1] - self.K 
                    else:
                        ex = self.K - stock_lattice[stock_last_row_idx-j,i-1]

                    early_exercise = np.maximum(ex, (1/rate) * (self.q * up_s + (1-self.q) * down_s))

                    if ex == early_exercise:
                        ex_early_lattice[last_row_idx-j,i-1] = 1
                    
                    lattice[last_row_idx-j,i-1] = early_exercise 
                else:
                    lattice[last_row_idx-j,i-1] = (1/rate) * (self.q * up_s + (1-self.q) * down_s)

        self.lattices[self.OPTION] = lattice
        self.lattices[self.EXERCISE] = ex_early_lattice

        return lattice

    def init_chooser_options_lattice(self):
        # Custom implementation
        size = self.n + 1
        lattice = np.zeros((size,size))
  
        stock_lattice = self.lattices[self.compute_on]
        stock_last_row_idx = stock_lattice.shape[0]

        lattice[:,-1] = stock_lattice[stock_last_row_idx-size:,size-1]

        last_row_idx = lattice.shape[0] - 1
        
        for i in reversed(range(0, self.n+1)):
            col = lattice[:,i]

            for j in range(i):
                down_s = col[last_row_idx-j]
                up_s = col[last_row_idx-(j+1)]

                lattice[last_row_idx-j,i-1] = (1/self.R) * (self.q * up_s + (1-self.q) * down_s)

        self.lattices[self.OPTION] = lattice

        return lattice

    @staticmethod
    def print_np_array(arr):
        print(pd.DataFrame(arr))
    
    def print_lattice(self,lat):
        self.print_np_array(self.lattices[lat])

    @staticmethod
    def resize_lattice(lattice, final_period):
        return lattice[lattice.shape[0]-final_period-1:,:final_period+1]

    def compute_forward_ZCB(self, forward_maturity):
        zcb = self.init_ZCB_lattice() 
        unit_zcb = self.init_ZCB_lattice(override_N = 1, override_maturity=forward_maturity)

        # Value of ZCB recieved at forward_maturity or at maturity of the bond is the same
        # as no coupons are paid at any point
        G0 = zcb[zcb.shape[0]-1, 0] / unit_zcb[unit_zcb.shape[0]-1,0]

        return (G0, zcb, unit_zcb)

    def compute_future_ZCB(self, future_maturity):
        zcb = self.init_ZCB_lattice()
        clipped_zcb = self.resize_lattice(zcb,future_maturity)
        no_discount_zcb = self.init_ZCB_lattice(override_N = clipped_zcb[:, future_maturity],
                                                override_maturity  = future_maturity,
                                                use_discounting=False)

        # Value of ZCB recieved at forward_maturity or at maturity of the bond is the same
        # as no coupons are paid at any point
        G0 = no_discount_zcb[no_discount_zcb.shape[0]-1, 0] 

        return (G0, no_discount_zcb)
    
    def init_ZCB_lattice(self, override_maturity=0, override_N=0, use_discounting=True):
        self.compute_option_parameters()
        self.override_computed_parameters()
        
        if override_maturity != 0:
            size = override_maturity + 1
        else:
            size = self.n_future + 1
            
        lattice = np.zeros((size,size))

        if type(override_N) == np.ndarray:
            lattice[:,-1] = override_N
        elif override_N != 0:
            lattice[:,-1] = np.full((lattice.shape[0],), override_N)
        else:
            lattice[:,-1] = np.full((lattice.shape[0],), self.N)  # Principle Returned

        short_rates = self.lattices[self.SHORT_RATE]

        last_row_idx = lattice.shape[0] - 1
        short_rates_last_row_idx = short_rates.shape[0] - 1 
        
        for i in reversed(range(1, size)):
            col = lattice[:,i]

            for j in range(i):
                down_s = col[last_row_idx-j]
                up_s = col[last_row_idx-(j+1)]

                if use_discounting:
                    lattice[last_row_idx-j,i-1] = 1/(1+short_rates[short_rates_last_row_idx-j,i-1])*(self.q * up_s + (1-self.q) * down_s)
                else:
                    lattice[last_row_idx-j,i-1] = (self.q * up_s + (1-self.q) * down_s)
                
        self.lattices[self.ZCB] = lattice

        return lattice

    def init_swap_lattice(self, forward_starting=0):
        self.compute_option_parameters()
        self.override_computed_parameters()
        
        size = self.n_future + 1
            
        lattice = np.zeros((size,size))
        short_rates = self.lattices[self.SHORT_RATE]
        
        lattice[:,-1] = (short_rates[:, lattice.shape[1]-1] - self.rK) / (1 + short_rates[:, lattice.shape[1]-1])

        last_row_idx = lattice.shape[0] - 1
        short_rates_last_row_idx = short_rates.shape[0] - 1 
        
        for i in reversed(range(1, size)):
            col = lattice[:,i] 

            for j in range(i):
                down_s = col[last_row_idx-j]
                up_s = col[last_row_idx-(j+1)]
                floating = short_rates[short_rates_last_row_idx-j,i-1]
                
                if i-1 < forward_starting:
                    lattice[last_row_idx-j,i-1] = 1/(1+floating)*(self.q * up_s + (1-self.q) * down_s)
                else:
                    lattice[last_row_idx-j,i-1] = 1/(1+floating)*((floating - self.rK) + self.q * up_s + (1-self.q) * down_s)

        lattice *= self.N 
        self.lattices[self.SWAP] = lattice

        return lattice
        

# Options Week 4
       
# Standard process
##bp = BinomialPricing(use_default = True)          
##bp.init_stock_lattice()
##bp.init_futures_lattice()
##bp.init_options_lattice()

# Compute Chooser Option
##bp = BinomialPricing(use_default = True)
##bp.compute_option_parameters()
##bp.init_stock_lattice()
##bp.init_futures_lattice()
##bp.option_type = bp.CALL
##call_lattice = bp.init_options_lattice()
##bp.option_type = bp.PUT
##put_lattice = bp.init_options_lattice()
##
##bp.n = 10 # Chooser expiration
##bp.T = bp.T * (bp.n / bp.n_future) # Correction for misallignment of expirations of stock / future and option
##bp.compute_option_parameters()
##
##bp.lattices[bp.CHOOSER] = np.maximum(call_lattice, put_lattice)[call_lattice.shape[0]-(bp.n+1):,:bp.n+1]
##bp.compute_on = bp.CHOOSER
##bp.init_chooser_options_lattice()

# Output
##if bp.is_american:
##    print(pd.DataFrame(bp.lattices[bp.EXERCISE]))
##
##print(pd.DataFrame(bp.lattices[bp.FUTURE]))
##print(pd.DataFrame(bp.lattices[bp.OPTION]))
##
##print(bp.lattices[bp.OPTION][-1,0])
##bp.print_lattice(bp.DEBUG)
##

# Rates Week 5
bp = BinomialPricing(use_default = True)
bp.init_short_rate_lattice()
##forward = bp.compute_forward_ZCB(4)
##bp.print_np_array(forward[1])
##bp.print_np_array(forward[2])
##print(forward[0])

##future = bp.compute_future_ZCB(4)
##bp.print_np_array(future[1])
##print(future[0])

##bp.init_ZCB_lattice()
##bp.init_options_lattice(use_short_rate=True)
##
### Output
##if bp.is_american:
##    print(pd.DataFrame(bp.lattices[bp.EXERCISE]))
##
##print(pd.DataFrame(bp.lattices[bp.ZCB]))
##print(pd.DataFrame(bp.lattices[bp.OPTION]))

##print(bp.lattices[bp.OPTION][-1,0])
bp.init_swap_lattice(forward_starting=1)
bp.init_options_lattice(use_short_rate=True)
##bp.print_lattice(bp.SHORT_RATE)
bp.print_lattice(bp.SWAP)
bp.print_lattice(bp.OPTION)

    
