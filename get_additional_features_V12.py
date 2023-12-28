import numpy as np
import pandas as pd


class Queue:
    def __init__(self, max_size=30):
    
        self.max_size = max_size
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        if len(self.items) < self.max_size:
            self.items.append(item)
        else:
            self.dequeue()
            self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            print("Queue is empty. Cannot dequeue.")

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        else:
            print("Queue is empty. Cannot peek.")

    def size(self):
        return len(self.items)
    
    def queue(self):
        return self.items

class DataFrameQueue:
    def __init__(self, max_size=30):
       
        self.max_size = max_size
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, dataframe):
        # Assuming "seconds" is the column indicating time
        if 'seconds_in_bucket' not in dataframe.columns:
            raise ValueError("DataFrame must have a 'seconds_in_bucket' column.")

        if dataframe['seconds_in_bucket'].nunique() == 1:
            if len(self.queue) < self.max_size:
                self.queue.append(dataframe)
            else:
                self.dequeue()
                self.queue.append(dataframe)
        else:
            raise ValueError("Error: 'seconds_in_bucket' column must have same values within the DataFrame.")

    def dequeue(self):
        if not self.is_empty():
            dequeued_item = self.queue.pop(0)
            return dequeued_item
        else:
            raise ValueError("Queue is empty. Cannot dequeue.")

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            raise ValueError("Queue is empty. Cannot peek.")

    def size(self):
        return len(self.queue)
    
    def queue(self):
        return self.queue
    
    def concated_df(self):
        return pd.concat(self.queue,ignore_index=True)
    
    def get_last_x_dataframes(self, x):
        return self.queue[-min(x, len(self.queue)):]# Ensure not to exceed the number of available DataFrames
    
    def get_last_x_dataframes_number(self, x):
        return min(x, len(self.queue)) # Ensure not to exceed the number of available DataFrames

class DataFrameQueue_interday:
    def __init__(self, max_size=30):
       
        self.max_size = max_size
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, dataframe):
        # Assuming "seconds" is the column indicating time
        if 'date_id' not in dataframe.columns:
            raise ValueError("DataFrame must have a 'date_id' column.")

        if dataframe['date_id'].nunique() == 1:
            if len(self.queue) < self.max_size:
                self.queue.append(dataframe)
            else:
                self.dequeue()
                self.queue.append(dataframe)
        else:
            raise ValueError("Error: 'date_id' column must have same values within the DataFrame.")

    def dequeue(self):
        if not self.is_empty():
            dequeued_item = self.queue.pop(0)
            return dequeued_item
        else:
            raise ValueError("Queue is empty. Cannot dequeue.")

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            raise ValueError("Queue is empty. Cannot peek.")

    def size(self):
        return len(self.queue)
    
    def queue(self):
        return self.queue
    
    def concated_df(self):
        return pd.concat(self.queue,ignore_index=True)
    
    def get_last_x_dataframes(self, x):
        return self.queue[-min(x, len(self.queue)):]# Ensure not to exceed the number of available DataFrames
    
    def get_last_x_dataframes_number(self, x):
        return min(x, len(self.queue)) # Ensure not to exceed the number of available DataFrames

class additional_features:
    def __init__(self,train_df):
        self.current_date = 8888
        self.current_second = None
        self.full_stock_id_list = train_df['stock_id'].unique()
        self.current_rolling_features_interday_df = pd.DataFrame()
        self.second_queue = Queue(30)
        self.market_relative_features_queue = DataFrameQueue(30)
        self.interday_features_queue = DataFrameQueue_interday(60)
        self.fake_interday_features_single_day_df = self.get_fake_interday_features_single_day_df(train_df)
        #for a smooth initialization, we first put a fake dataframe into queue
        self.interday_features_queue.enqueue(self.fake_interday_features_single_day_df)
        self.get_rolling_features_interday()

        self.original_features_queue = DataFrameQueue(60)

    def check_fake_interday_features_single_day_df(self):

        return self.fake_interday_features_single_day_df

    def load_data(self,df):

        self.original_features = self.fillnavalue(df)
        self.Market_relative_features = pd.DataFrame()
        self.rolling_features = pd.DataFrame()
        self.rolling_market_relative_features_intraday = pd.DataFrame()
        self.market_idx = {}  
        self.result_df = pd.DataFrame()

        # get the current date 
        if self.original_features['date_id'].nunique() != 1:
            raise ValueError("Error: Duplicate values found in 'date_id' column.")
        else: 
            
            if self.original_features['date_id'].iloc[0] != self.current_date and self.current_date != 8888: 
                
                #Initialize or Date changes, re-initialization of DataFrameQueue and second Queue
                self.current_date = self.original_features['date_id'].iloc[0]
                self.current_second = None

                #compute interday features when day is finished
                self.date_change()
                
                #reset all Queue
                self.original_features_queue = DataFrameQueue(30)
                self.market_relative_features_queue = DataFrameQueue(30)
                self.second_queue = Queue(30)

            else:  
                self.current_date = self.original_features['date_id'].iloc[0]


        # get the current second
        if df['seconds_in_bucket'].nunique() != 1:
            raise ValueError("Error: Duplicate values found in 'seconds_in_bucket' column.")
        else:

            if df['seconds_in_bucket'].iloc[0] != self.current_second: 
                
                self.current_second = df['seconds_in_bucket'].iloc[0]
                self.second_queue.enqueue(self.current_second)
                self.original_features_queue.enqueue(self.original_features)
                
            else:  
                raise ValueError("Error: same 'seconds_in_bucket' is already exist")
            
        self.compute()
    
    def date_change(self):

        #put original_features_queue (with full 540 seconds) into compute_interday_features_single_day to compute interday_features_single_day_df
        interday_features_single_day_df = self.compute_interday_features_single_day(self.original_features_queue.concated_df())
        interday_features_single_day_df_filled =  self.fill_full_stock_ids(interday_features_single_day_df)
        self.interday_features_queue.enqueue(interday_features_single_day_df_filled)
        
        #refreash the current_rolling_features_interday_df
        self.get_rolling_features_interday()
      
    def get_fake_interday_features_single_day_df(self,df):
        
        # 加载数据
        df = self.fillnavalue(df)

        # 初始化一个空的 DataFrame 用于存储最终结果
        interday_features = pd.DataFrame()
        
        # 对于 date_id 从 0 到 x，计算每个 date_id 的 diffs 并拼接
        for date_id in df['date_id'].unique():
            # 筛选特定 date_id 的数据
            date_df = df.loc[df["date_id"] == date_id]

            # 计算当前 date_id 的 diffs
            diffs = self.compute_interday_features_single_day(date_df)

            # 确保 diffs 是 DataFrame 类型
            if isinstance(diffs, pd.DataFrame):
                # 将结果拼接到最终的 DataFrame
                interday_features = pd.concat([interday_features, diffs], ignore_index=True)
            else:
                print(f"Date ID {date_id} did not return a DataFrame.")

        # 选择除 'date' 之外的所有列
        columns_except_date = interday_features.columns.drop('date_id')

        fake_interday_features_single_day_df = interday_features[columns_except_date].groupby('stock_id').mean()
        fake_interday_features_single_day_df["date_id"] = 8888
        # 重置索引，使得列名都在同一行
        fake_interday_features_single_day_df = fake_interday_features_single_day_df.reset_index()


        return fake_interday_features_single_day_df

    def check_original_features_queue(self):

        return self.original_features_queue.queue

    def get_market_idx(self):
    
        #Input: a DataFrame including all original features
        #Output: a dict including all additional features
        train_single_sec =self.original_features
        Market_idx = {} 
                
        #Auction book data
        all_imbalance_size = train_single_sec["imbalance_size"]
        all_imbalance_buy_sell_flag	= train_single_sec["imbalance_buy_sell_flag"]
        all_matched_size = train_single_sec["matched_size"]
        all_far_price = train_single_sec["far_price"]

        #Continues Order book data
        all_bid_price = train_single_sec["bid_price"]
        all_ask_price = train_single_sec["ask_price"]
        all_bid_size = train_single_sec["bid_size"]
        all_ask_size = train_single_sec["ask_size"]
        all_wap = train_single_sec["wap"]

        #Spread in bps
        all_spread = (all_ask_price - all_bid_price) / (all_ask_price + all_bid_price) * 2* 100

        #Combine book 
        all_near_price = train_single_sec["near_price"]
        all_reference_price	= train_single_sec["reference_price"]

        #Size of all orders in Auction book, Continues Order book and in both books (in dollors)
        all_auction_book_size = all_matched_size + all_imbalance_size 
        all_continues_book_size = all_bid_size + all_ask_size
        all_combined_book_size = all_auction_book_size + all_continues_book_size



        #Market features of Auction book
        Market_imbalance_ratio = all_imbalance_size.sum() / all_auction_book_size.sum()
        Market_matched_ration = all_matched_size.sum() / all_auction_book_size.sum()
        
        if all_imbalance_size.sum() == 0:
            Market_imbalance_direction = 0
        else:
            Market_imbalance_direction = (all_imbalance_buy_sell_flag*all_imbalance_size).sum() / all_imbalance_size.sum()

        Market_far_price = (all_far_price*all_auction_book_size).sum() / all_auction_book_size.sum()

        Market_idx["Market_imbalance_ratio"] = Market_imbalance_ratio
        Market_idx["Market_matched_ratio"] = Market_matched_ration
        Market_idx["Market_imbalance_direction"] = Market_imbalance_direction
        Market_idx["Market_far_price"] = Market_far_price
        

        #Market features of Continues Order book 
        Market_bid_price = (all_bid_price*all_bid_size).sum() / all_bid_size.sum()
        Market_ask_price = (all_ask_price*all_ask_size).sum() / all_ask_size.sum()
        
        Market_avg_bid_size = all_bid_size.sum()/all_bid_size.count()
        Market_avg_ask_size = all_ask_size.sum()/all_ask_size.count()

        Market_bid_ratio = all_bid_size.sum() / all_continues_book_size.sum()
        Market_ask_ratio = all_ask_size.sum() / all_continues_book_size.sum()

        Market_Spread = (all_spread*all_continues_book_size).sum() / all_continues_book_size.sum()
        Market_wap = (all_wap*all_continues_book_size).sum() / all_continues_book_size.sum()

        Market_idx["Market_bid_price"] = Market_bid_price
        Market_idx["Market_ask_price"] = Market_ask_price
        Market_idx["Market_bid_ratio"] = Market_bid_ratio
        Market_idx["Market_ask_ratio"] = Market_ask_ratio
        
        Market_idx["Market_avg_bid_size"] = Market_avg_bid_size
        Market_idx["Market_avg_ask_size"] = Market_avg_ask_size
        
        
        Market_idx["Market_spread"] = Market_Spread
        Market_idx["Market_wap"] = Market_wap

        #Market features of Combine book 
        
        Market_near_price = (all_near_price*all_combined_book_size).sum() / all_combined_book_size.sum()
        Market_reference_price = (all_reference_price*all_combined_book_size).sum() / all_combined_book_size.sum()

        Market_idx["Market_near_price"] = Market_near_price
        Market_idx["Market_reference_price"] = Market_reference_price


        self.market_idx = Market_idx

    def check_market_idx(self):

        train_with_Market_relative_features = pd.DataFrame()
        train_with_Market_relative_features["stock_id"] = train["stock_id"]
        train_with_Market_relative_features["date_id"] = train["date_id"]
        train_with_Market_relative_features["seconds_in_bucket"] = train["seconds_in_bucket"]
        train_with_Market_relative_features["row_id"] = train["row_id"]

        return self.market_idx

    def get_market_relative_features(self):

        Market_idx_df = self.market_idx
        train = self.original_features

        features_column =["Market_imbalance_ratio", "Market_matched_ratio",  "Market_imbalance_direction",
                        "Market_far_price",       "Market_bid_price",      "Market_ask_price",           "Market_bid_ratio",   "Market_ask_ratio", "Market_avg_bid_size",
                        "Market_avg_ask_size",    "Market_spread",         "Market_wap",                 "Market_near_price",  "Market_reference_price"]

        #multi_index = pd.MultiIndex.from_product([max_date_id, max_seconds_in_bucket],
                                                #names=['date_id', 'seconds_in_bucket'])
        train_with_Market_relative_features = pd.DataFrame()
        train_with_Market_relative_features["stock_id"] = train["stock_id"]
        train_with_Market_relative_features["date_id"] = train["date_id"]
        train_with_Market_relative_features["seconds_in_bucket"] = train["seconds_in_bucket"]
        train_with_Market_relative_features["row_id"] = train["row_id"]
        

        for column in features_column:
            train_with_Market_relative_features[column] = Market_idx_df[column]

        
        #Add feature "spread"
        train_with_Market_relative_features["spread"]= (train["ask_price"] - train["bid_price"]) / (train["ask_price"] + train["bid_price"]) * 2 * 100

        #Add feature "imbalance/matched ratio"
        train_with_Market_relative_features["imbalance_ratio"] = train["imbalance_size"] / (train["imbalance_size"] + train["matched_size"])
        train_with_Market_relative_features["matched_ratio"] = train["matched_size"] / (train["imbalance_size"] + train["matched_size"])

        #Add feature "bid/ask ratio"
        train_with_Market_relative_features["bid_ratio"] = train["bid_size"] / (train["bid_size"] + train["ask_size"])
        train_with_Market_relative_features["ask_ratio"] = train["ask_size"] / (train["bid_size"] + train["ask_size"])

        #Add feature "continue book intention" and "auction book intention"
        train_with_Market_relative_features["continue_book_intention"] = train_with_Market_relative_features["bid_ratio"] - train_with_Market_relative_features["ask_ratio"]
        train_with_Market_relative_features["auction_book_intention"] = train["imbalance_buy_sell_flag"] * train_with_Market_relative_features["imbalance_ratio"] 

        #Add feature "Resonance of both book"
        train_with_Market_relative_features["Resonance_both_book"] = train_with_Market_relative_features["continue_book_intention"] * train_with_Market_relative_features["auction_book_intention"]


        #Since data are all at same date and same second_in_bucket, condition is not necessery
        condition = True     
        
        #Auction book Features   
        
        #Add feature "imbalance/matched ratio" 
        train_with_Market_relative_features["relative_imbalance_ratio"] = train_with_Market_relative_features["imbalance_ratio"]/Market_idx_df["Market_imbalance_ratio"]
        train_with_Market_relative_features["relative_matched_ratio"] = train_with_Market_relative_features["matched_ratio"]/Market_idx_df["Market_matched_ratio"]
        train_with_Market_relative_features["relative_far_price"] = train['far_price']/Market_idx_df["Market_far_price"]
        
        #Add feature "Resonance with Market" 
        train_with_Market_relative_features["Resonance_auction_book_market"] = train_with_Market_relative_features["auction_book_intention"] * Market_idx_df["Market_imbalance_direction"]
        
        #Add feature "spread"
        train_with_Market_relative_features["relative_spread"] = train_with_Market_relative_features["spread"]/Market_idx_df["Market_spread"]
        
        #Continues book Features
        train_with_Market_relative_features["Resonance_continue_book_market"] = train_with_Market_relative_features['continue_book_intention'] / (Market_idx_df["Market_bid_ratio"] - Market_idx_df["Market_ask_ratio"])
        train_with_Market_relative_features['relative_bid_price'] = train['bid_price']/Market_idx_df["Market_bid_price"]
        train_with_Market_relative_features['relative_ask_price'] = train['ask_price']/Market_idx_df["Market_ask_price"]  
        train_with_Market_relative_features['relative_bid_ratio'] = train_with_Market_relative_features['bid_ratio']/Market_idx_df["Market_bid_ratio"]
        train_with_Market_relative_features['relative_ask_ratio'] = train_with_Market_relative_features['ask_ratio']/Market_idx_df["Market_ask_ratio"]
        
        train_with_Market_relative_features['relative_wap']       = train['wap']/Market_idx_df["Market_wap"]
        
        train_with_Market_relative_features["relative_bid_size"]  = train['bid_size']/Market_idx_df["Market_avg_bid_size"]
        train_with_Market_relative_features["relative_ask_size"]  = train['ask_size']/Market_idx_df["Market_avg_ask_size"]
        
        #Combine book Features 
        train_with_Market_relative_features["relative_near_price"] = train["near_price"]/Market_idx_df["Market_near_price"]
        train_with_Market_relative_features["relative_reference_price"] = train["reference_price"]/Market_idx_df["Market_reference_price"]

        self.Market_relative_features = train_with_Market_relative_features.reset_index(drop=True)
        self.market_relative_features_queue.enqueue(self.Market_relative_features)

    def check_Market_relative_features(self):

        return self.Market_relative_features

    def get_rolling_features_intraday(self):

        #train = train.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket'])
        train = self.original_features
        rolling_features_intraday_df = pd.DataFrame()
     
        rolling_features_intraday_df["stock_id"] = train["stock_id"]
        rolling_features_intraday_df["date_id"] = train["date_id"]
        rolling_features_intraday_df["seconds_in_bucket"] = train["seconds_in_bucket"]
        rolling_features_intraday_df["row_id"] = train["row_id"]
        
        

        window_sizes = [2, 4, 8, 16, 32]
        indexing_columns = ['stock_id', 'date_id', 'seconds_in_bucket', 'row_id', 'time_id','target']
        #grouped_df = train.groupby(['stock_id', 'date_id'], sort=False)  

        # go through window_sizes
        for window_size in window_sizes:
            print("get_rolling_features_intraday: window_size",window_size*10)
                  
            # Get the last x DataFrames from the queue or all DataFrames if fewer than x
            last_x_dataframes = self.original_features_queue.get_last_x_dataframes(window_size*10)

            # Concatenate the last x DataFrames
            concatenated_df = pd.concat(last_x_dataframes, ignore_index=True)
            print(concatenated_df)

            for column in train.columns[2:]:
                print("get_rolling_features_intraday: column",column)

                if column not in indexing_columns:

                    rolling_features_intraday_df[f'intraday_avg_{column}_{window_size*10}'] = concatenated_df.groupby('stock_id')[column].mean().fillna(0).values

                    rolling_features_intraday_df[f'intraday_std_{column}_{window_size*10}'] = concatenated_df.groupby('stock_id')[column].std().fillna(0).values


        self.rolling_features = rolling_features_intraday_df.reset_index(drop=True)

    def check_rolling_features(self):

        return self.rolling_features

    def get_rolling_features_interday(self):

        #train = train.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket'])

        rolling_features_interday = pd.DataFrame()
     
        rolling_features_interday["stock_id"] = self.full_stock_id_list
        rolling_features_interday["date_id"] = self.current_date
        
        window_sizes = [2, 4, 8, 16, 32]
        indexing_columns = ['stock_id', 'date_id', 'seconds_in_bucket', 'row_id', 'time_id','target']

        delta_columns =   ["wap_delta","reference_price_delta","far_price_delta","near_price_delta","bid_price_delta","ask_price_delta"]
        delta_u_columns = ["wap_delta_u","reference_price_delta_u","far_price_delta_u","near_price_delta_u","bid_price_delta_u","ask_price_delta_u"]
        delta_d_columns = ["wap_delta_d","reference_price_delta_d","far_price_delta_d","near_price_delta_d","bid_price_delta_d","ask_price_delta_d"]


        #grouped_df = train.groupby(['stock_id', 'date_id'], sort=False)  

        # go through window_sizes
        for window_size in window_sizes:
            print("get_rolling_features_interday: window_size",window_size)
                  
            # Get the last x DataFrames from the queue or all DataFrames if fewer than x
            last_x_dataframes = self.interday_features_queue.get_last_x_dataframes(window_size)

            # Concatenate the last x DataFrames
            concatenated_df = pd.concat(last_x_dataframes, ignore_index=True)
            #print(concatenated_df)

            for column in concatenated_df.columns[2:]:
                print("get_rolling_features_interday: column",column)

                if column not in indexing_columns and column not in delta_u_columns and column not in delta_d_columns:

                    if column in delta_columns:

                        rolling_features_interday[f'interday_rasing_ratio_{column}_{window_size}'] = concatenated_df.groupby('stock_id')[column].mean().fillna(0).values
                        rolling_features_interday[f'interday_rsi_u_{column}_{window_size}'] = concatenated_df.groupby('stock_id')[f'{column}_u'].mean().fillna(0).values
                        rolling_features_interday[f'interday_rsi_d_{column}_{window_size}'] = concatenated_df.groupby('stock_id')[f'{column}_d'].mean().fillna(0).values
                        rolling_features_interday[f'interday_rsi_{column}_{window_size}'] = rolling_features_interday[f'interday_rsi_u_{column}_{window_size}']/rolling_features_interday[f'interday_rsi_d_{column}_{window_size}'] 

                    else:
                        rolling_features_interday[f'interday_avg_{column}_{window_size}'] = concatenated_df.groupby('stock_id')[column].mean().fillna(0).values
                        rolling_features_interday[f'interday_std_{column}_{window_size}'] = concatenated_df.groupby('stock_id')[column].std().fillna(0).values


        self.current_rolling_features_interday_df = rolling_features_interday.reset_index(drop=True)
    
    def get_rolling_market_relative_features_intraday(self):

        #train = train.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket'])
        train = self.Market_relative_features
        rolling_features_intraday_df = pd.DataFrame()
     
        rolling_features_intraday_df["stock_id"] = train["stock_id"]
        rolling_features_intraday_df["date_id"] = train["date_id"]
        rolling_features_intraday_df["seconds_in_bucket"] = train["seconds_in_bucket"]
        rolling_features_intraday_df["row_id"] = train["row_id"]
        
        window_sizes = [2, 4, 8, 16, 32]
        indexing_columns = ['stock_id', 'date_id', 'seconds_in_bucket', 'row_id', 'time_id','target']
        #grouped_df = train.groupby(['stock_id', 'date_id'], sort=False)  

        # go through window_sizes
        for window_size in window_sizes:
            print("get_market_relative_features_intraday: window_size",window_size)
                  
            # Get the last x DataFrames from the queue or all DataFrames if fewer than x
            last_x_dataframes = self.market_relative_features_queue.get_last_x_dataframes(window_size)

            # Concatenate the last x DataFrames
            concatenated_df = pd.concat(last_x_dataframes, ignore_index=True)
            print(concatenated_df)

            for column in train.columns[2:]:
                print("get_rolling_features_intraday: column",column)

                if column not in indexing_columns:

                    rolling_features_intraday_df[f'intraday_market_avg_{column}_{window_size*10}'] = concatenated_df.groupby('stock_id')[column].mean().fillna(0).values

                    rolling_features_intraday_df[f'intraday_market_std_{column}_{window_size*10}'] = concatenated_df.groupby('stock_id')[column].std().fillna(0).values


        self.rolling_market_relative_features_intraday = rolling_features_intraday_df.reset_index(drop=True)

    def check_rolling_market_relative_features_intraday(self):

        return self.rolling_market_relative_features_intraday

    def fillnavalue(self,df):
    
        fill_content= {"imbalance_size":0,
        "imbalance_buy_sell_flag":0,
        "reference_price":1,
        "matched_size":0,
        "far_price":1,
        "near_price":1,
        "bid_price":1,	
        "bid_size":0,	
        "ask_price":1,
        "ask_size":0,	
        "wap":1}

        for column, fill_value in fill_content.items():

            if column in df:

                df[column].fillna(fill_value, inplace=True)

        return df
    
    def fill_full_stock_ids(self,df):
        interday_features_single_day_df = df
        self.fake_interday_features_single_day_df
        # 合并两个 DataFrame，保留所有行
        interday_features_single_day_df_filled = pd.merge(interday_features_single_day_df, self.fake_interday_features_single_day_df, 
                                                        on='stock_id', suffixes=('', '_fake'), how='outer', indicator=True)

        # 对于每列，使用 interday_features_single_day_df 的值，如果缺失则使用 fake_interday_features_single_day_df 的值
        for col in interday_features_single_day_df.columns:
            if col != 'stock_id':  # 对于除 'stock_id' 之外的所有列
                interday_features_single_day_df_filled[col] = interday_features_single_day_df_filled.apply(
                    lambda row: row[col] if not pd.isna(row[col]) else row[col + '_fake'], axis=1)

        # 删除所有带 '_fake' 后缀的列和 '_merge' 列
        interday_features_single_day_df_filled = interday_features_single_day_df_filled.drop(
            columns=[col for col in interday_features_single_day_df_filled.columns if '_fake' in col] + ['_merge'])

        # # 替换 'date_id' 列中的 8888 为 interday_features_single_day_df 中的 'date_id' 值
        real_date_id = interday_features_single_day_df['date_id'].iloc[0]
        interday_features_single_day_df_filled['date_id'] = interday_features_single_day_df_filled['date_id'].replace(8888, real_date_id)

        # 保证 date_id 列是整数类型
        interday_features_single_day_df_filled['date_id'] = interday_features_single_day_df_filled['date_id'].astype(int)

        # 按照 stock_id 对所有行排序
        interday_features_single_day_df_filled = interday_features_single_day_df_filled.sort_values(by='stock_id')
        interday_features_single_day_df_filled.index = interday_features_single_day_df_filled['stock_id']

        return interday_features_single_day_df_filled

    def compute(self):

        self.get_market_idx()
        self.get_rolling_features_intraday()
        self.get_market_relative_features()
        self.get_rolling_market_relative_features_intraday()

        self.result_df = self.original_features
        self.result_df = self.result_df.merge(self.Market_relative_features, how='left', on= ["stock_id","date_id","seconds_in_bucket","row_id"])
        self.result_df = self.result_df.merge(self.rolling_features, how='left', on= ["stock_id","date_id","seconds_in_bucket","row_id"])
        self.result_df = self.result_df.merge(self.rolling_market_relative_features_intraday, how='left', on= ["stock_id","date_id","seconds_in_bucket","row_id"])
    
    def compute_interday_features_single_day(self,df):

        #input: 108000(all stock* all seconds)*20(features)
        #output: 200(all stock)*(20*x)(features of all features)

        # 初始化列表用于存储结果
        diff_rows = []

        # 定义两组特征列
        full_feature_columns = ['reference_price', 'far_price', 'near_price', 'bid_price', 'ask_price', 'wap']
        basic_feature_columns = ['imbalance_size', 'imbalance_buy_sell_flag', 'matched_size', 'bid_size', 'ask_size']

        # 对每支股票进行遍历
        for stock_id, group in df.groupby('stock_id'):
            # 初始化临时字典来存储结果
            result_dict = {'stock_id': stock_id, 'date_id': group['date_id'].iloc[0]}

            # 全量计算特定特征
            for feature in full_feature_columns:
                max_value = group[feature].max()
                min_value = group[feature].min()
                delta = group[feature].iloc[-1] - group[feature].iloc[0]

                result_dict[f'{feature}_max_min_diff'] = max_value - min_value
                result_dict[f'{feature}_delta'] = delta
                result_dict[f'{feature}_direction'] = 1 if delta > 0 else 0
                result_dict[f'{feature}_std'] = group[feature].std()
                result_dict[f'{feature}_avg'] = group[feature].mean()
                result_dict[f'{feature}_delta_u'] = delta if delta > 0 else 0
                result_dict[f'{feature}_delta_d'] = abs(delta) if delta < 0 else 0

            # 基础计算其他特征
            for feature in basic_feature_columns:
                max_value = group[feature].max()
                min_value = group[feature].min()

                result_dict[f'{feature}_max_min_diff'] = max_value - min_value
                result_dict[f'{feature}_std'] = group[feature].std()
                result_dict[f'{feature}_avg'] = group[feature].mean()

            # 将结果添加到列表中
            diff_rows.append(result_dict)

        # 从列表创建 DataFrame
        interday_features_single_day_df = pd.DataFrame(diff_rows)

        return interday_features_single_day_df
  
    def intraday_output(self):

        return self.result_df
    
    def interday_output(self):

        return self.current_rolling_features_interday_df

# How to use
# 1. Initialize the Intraday Class when the whole train_df will be needed: additional_df = additional_features(train_df)
# 2. Feed the data using: additional_df.load_data(df)
#    - Be careful: The data should contain a single date_id and seconds_in_bucket.
# 3. Get the processed output using: 
#    - intraday features: additional_df.intraday_output()
#    - interday features: additional_df.interday_output()
#    - interday features will be updated only when day was finished indicating by changing of date_id
#    - a date_id = 8888 means that there is no previous day, and the output data was simply avg over whole 480 days 


#Example

train = pd.read_csv("/Users/xinhang.li/Documents/kaggle/optiver-trading-at-the-close/train.csv")
additional_df = additional_features(train)
#additional_df.check_fake_interday_features_single_day_df()


train_3_00 = train.loc[(train["date_id"] == 3) & (train["seconds_in_bucket"] == 0) ]
train_3_10 = train.loc[(train["date_id"] == 3) & (train["seconds_in_bucket"] == 10) ]
train_3_20 = train.loc[(train["date_id"] == 3) & (train["seconds_in_bucket"] == 20) ]
train_3_30 = train.loc[(train["date_id"] == 3) & (train["seconds_in_bucket"] == 30) ]
#train_3_00.to_csv("test_original_train_3_00.csv")

train_4_00 = train.loc[(train["date_id"] == 4) & (train["seconds_in_bucket"] == 0) ]
train_4_10 = train.loc[(train["date_id"] == 4) & (train["seconds_in_bucket"] == 10) ]
train_4_20 = train.loc[(train["date_id"] == 4) & (train["seconds_in_bucket"] == 20) ]
train_4_30 = train.loc[(train["date_id"] == 4) & (train["seconds_in_bucket"] == 30) ]
#train_4_00.to_csv("test_original_train_4_00.csv")

additional_df.load_data(df=train_3_00)
additional_df.intraday_output().to_csv("test_intraday_output_3_00.csv")
additional_df.interday_output().to_csv("test_interday_output_3.csv")

additional_df.load_data(df=train_3_10)
#additional_df.intraday_output().to_csv("test_intraday_output_3_10.csv")

additional_df.load_data(df=train_3_20)
#additional_df.intraday_output().to_csv("test_intraday_output_3_20.csv")

additional_df.load_data(df=train_3_30)
#additional_df.intraday_output().to_csv("test_intraday_output_3_30.csv")



#Now date_id changes from 3 to 4
additional_df.load_data(df=train_4_00)
additional_df.intraday_output().to_csv("test_intraday_output_4_00.csv")
additional_df.interday_output().to_csv("test_interday_output_4.csv")



""" 
date_id = 0
combined_df = pd.DataFrame()
for seconds in train['seconds_in_bucket'].unique():
    # Filter the train DataFrame for the current date_id and seconds_in_bucket
    subset = train.loc[(train["date_id"] == date_id) & (train["seconds_in_bucket"] == seconds)]

    # Load the data to intraday_df and process it
    intraday_df.load_data(df=subset)
    result = intraday_df.output()

    # Append the result to the combined DataFrame
    combined_df = pd.concat([combined_df, result], ignore_index=True) 
"""

