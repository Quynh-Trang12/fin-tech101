## COS30018 - Option C - Task 5: Machine Learning 2

After you have completed Task 4, our code base has moved to version v0.4 . Our current program only uses one feature (e.g. the closing stock price of a company) to predict its closing price for a single day in the future. We will now try to solve more advanced prediction problems including multivariate prediction and multistep prediction. The multistep prediction problem applies to time series forecasting that requires a prediction of multiple time steps into the future (e.g. the closing stock prices of the company in multiple days into the future). The multivariate prediction problem on the other hand would take multiple time series as the input for the machine learning model to make the prediction. In its simplest form, the multivariate prediction problem for our project would just take the different features (such as opening price, highest price, lowest price, closing price, adjusted closing price, trading volume) as the input time series for making the prediction of the closing price of the company in the future. The more general multivariate prediction problem can also take other time series as the input to the model (for instance, the time series of the closing prices of the related companies in the same sector, or the time series of the market index).

## Your tasks this week:

1. Implement a function that solve the multistep prediction problem to allow the prediction to be made for a sequence of closing prices of k days into the future.
2. Implement a function that solve the simple multivariate prediction problem to that takes into account the other features for the same company (including opening price, highest price, lowest price, closing price, adjusted closing price, trading volume) as the input for predicting the closing price of the company for a specified day in the future.
3. Combine the above two functions to solve the multivariate, multistep prediction problem.
4. Upload your Task 5 Report (as a PDF file) to the project Wiki before the deadline and email your project leader to notify that it is ready for viewing and feedback.

## Your Task 5 Report will contain the following details:

- Summary of your effort to implement the functions specified above and explain the less straightforward lines of code, focusing especially on those lines that require you to do some research on the Internet (with proper references to the online resources you used).
- Summaries of the results of your experiments with solving the multivariate and multistep prediction problems.

## Due date: 11:59pm Sunday 28 September 2024

## Assessment Criteria:

You can get up to 15 marks for successfully completing Task C.5.