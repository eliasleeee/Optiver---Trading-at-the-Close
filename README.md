# Optiver-Trading-at-the-Close
This repo is associated with match hosted by Kaggle&Optiver 
Website:https://www.kaggle.com/competitions/optiver-trading-at-the-close

We use TFT model in model.ipynb
We also use some additional features (about 1000 features) to improve performance, you can find them in get_additional_features_V12.py, as well as the describtion of them in features_description.xlsx 

Here are some examples:

++++++++++++++++++++++++++++++++
Train_market_relative	提取的Features 数据集，Index与原始训练数据集对应	
		
Axis Index	   各种Index 	
stock_id	   股票代码	        0 - 199
date_id	           日期代码	        day 0 -  day 480
seconds_in_bucket  秒代码	                0s - 540s (every 10 s)
row_id	          （冗余）Row 标识码	0_0_0 - 480_540_199
	
++++++++++++++++++++++++++++++++	
Continues book (C book) features	正常交易 Book Features	
spread	                买价和卖价的价差	                 ～0 AND > 0
relative_spread	        买价和卖价的价差 / 大盘平均价差	
relative_bid_price	买价 / 大盘平均买价	         ～ 1
relative_ask_price	卖价 / 大盘平均卖价	         ～ 1
bid_ratio	        买单量/总单量	               0 ～ 1
ask_ratio	        卖单量/总单量	               0 ～ 1
relative_bid_ratio	买单比例/大盘买单比例	
relative_ask_ratio	卖单比例/大盘卖单比例	
relative_wap	        中间价/大盘中间价	                 ～ 1
relative_far_price	瞬时配对价/大盘瞬时配对价	         ～ 1
relative_bid_size	买单量/大盘总买单量	
relative_ask_size	卖单量/大盘总卖单量	
continue_book_intention（买单量-卖单量）/总单量	      - 1 ～ 1
Resonance_continue_book_market	continue_book_intention*大盘平均continue_book_intention	
	
	
++++++++++++++++++++++++++++++++
Auction book (A book) features	 拍卖 Book Features	
imbalance_ratio	                 未配对Orders 的Size / 总Orders的Size	             0 ～ 1
matched_ratio	                 已配对Orders 的Size / 总Orders的Size	             0 ～ 1
relative_imbalance_ratio	 imbalance_ratio/imbalance_ratio的大盘平均值	
relative_matched_ratio	         matched_ratio/matched_ratio的大盘平均值	
auction_book_intention	         imbalance_buy_sell_flag*imbalance_ratio	   - 1 ～ 1
Resonance_auction_book_market	 auction_book_intention*大盘平均continue_book_intention	


++++++++++++++++++++++++++++++++		
Combine of both bpok	         两本Book的共同Features	
relative_near_price	         near_price/near_price的大盘平均值	
relative_reference_price	 reference_price/reference_price的大盘平均值	
Resonance_both_book	         Resonance_continue_book_market*Resonance_auction_book_market	- 1 ～ 1
