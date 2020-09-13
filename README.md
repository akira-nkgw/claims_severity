# Allstate Claims Severity- Kaggle Challange

## Introduction
This project is to develop automated methods of predicting the cost, and hence severity, of claims. And this is proposed by AllState, an insurance company.
The mission of the project is ultimately to be able to find more severe claims from many claims reports. So the claims adjuster can help those people who needs
immediate help will be able to be paid. <br> <br>
"When you’ve been devastated by a serious car accident, your focus is on the things that matter the most: family, friends, and other loved ones. Pushing paper with your insurance agent is the last place you want your time or mental energy spent. This is why Allstate, a personal insurer in the United States, is continually seeking fresh ideas to improve their claims service for the over 16 million households they protect."

## Data Source
I got this data source from Kaggle.<br> https://www.kaggle.com/c/allstate-claims-severity/   <br>
Each row in this dataset represents an insurance claim. You must predict the value for the 'loss' column. Variables prefaced with 'cat' are categorical, while those prefaced with 'cont' are continuous.

## Data Cleansing - missing value handling
train.csv has 188319 rows and 132 columns of data. I noticed that there are missing values in the dataset as below. <br>
{'id': 0, 'cat1': 0, 'cat2': 0, 'cat3': 0, 'cat4': 0, 'cat5': 0, 'cat6': 0, 'cat7': 0, 'cat8': 0, 'cat9': 0, 'cat10': 0, 'cat11': 0, 'cat12': 1, 'cat13': 1, 'cat14': 1, 'cat15': 1, 'cat16': 1, 'cat17': 1, 'cat18': 1, 'cat19': 1, 'cat20': 1, 'cat21': 1, 'cat22': 1, 'cat23': 1, 'cat24': 1, 'cat25': 1, 'cat26': 1, 'cat27': 1, 'cat28': 1, 'cat29': 1, 'cat30': 1, 'cat31': 1, 'cat32': 1, 'cat33': 1, 'cat34': 1, 'cat35': 1, 'cat36': 1, 'cat37': 1, 'cat38': 1, 'cat39': 1, 'cat40': 1, 'cat41': 1, 'cat42': 1, 'cat43': 1, 'cat44': 1, 'cat45': 1, 'cat46': 1, 'cat47': 1, 'cat48': 1, 'cat49': 1, 'cat50': 1, 'cat51': 1, 'cat52': 1, 'cat53': 1, 'cat54': 1, 'cat55': 1, 'cat56': 1, 'cat57': 1, 'cat58': 1, 'cat59': 1, 'cat60': 1, 'cat61': 1, 'cat62': 1, 'cat63': 1, 'cat64': 1, 'cat65': 1, 'cat66': 1, 'cat67': 1, 'cat68': 1, 'cat69': 1, 'cat70': 1, 'cat71': 1, 'cat72': 1, 'cat73': 1, 'cat74': 1, 'cat75': 1, 'cat76': 1, 'cat77': 1, 'cat78': 1, 'cat79': 1, 'cat80': 1, 'cat81': 1, 'cat82': 1, 'cat83': 1, 'cat84': 1, 'cat85': 1, 'cat86': 1, 'cat87': 1, 'cat88': 1, 'cat89': 1, 'cat90': 1, 'cat91': 1, 'cat92': 1, 'cat93': 1, 'cat94': 1, 'cat95': 1, 'cat96': 1, 'cat97': 1, 'cat98': 1, 'cat99': 1, 'cat100': 1, 'cat101': 1, 'cat102': 1, 'cat103': 1, 'cat104': 1, 'cat105': 1, 'cat106': 1, 'cat107': 1, 'cat108': 1, 'cat109': 1, 'cat110': 1, 'cat111': 1, 'cat112': 1, 'cat113': 1, 'cat114': 1, 'cat115': 1, 'cat116': 1, 'cont1': 1, 'cont2': 1, 'cont3': 1, 'cont4': 1, 'cont5': 1, 'cont6': 1, 'cont7': 1, 'cont8': 1, 'cont9': 1, 'cont10': 1, 'cont11': 1, 'cont12': 1, 'cont13': 1, 'cont14': 1, 'loss': 1}
<br>
I decided to simply delete the missing value because I suspect probably only one row has missing values for many columns. After dropping N/A values from the dataset, the number of rows became 188318, which clearly shows that one row had many missing values. I cleaned the dataset on <b>'clean_data.ipynb'</b> and produced a new csv file for clean dataset, <b>'train_clean.csv'</b>. 

## Exploratory data analysis (EDA)


