
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import statsmodels.api as sm  # For OLS trendline in Plotly
import ast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the data and model
with open('model/data.df', 'rb') as file:
    # File handling code here', 'rb') as file:
    df = pickle.load(file)
with open('data/target', 'rb') as file:
    # File handling code here', 'rb') as file:
    target= pickle.load(file)
with open('model/model(random_forest).pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Sidebar Navigation
st.sidebar.title("Real Estate Dashboard Gurgaon")
selection = st.sidebar.radio("Go to", ["Home", "Price Prediction", "Analytical Module", "Insight Model", "Recommender System"])
    # Home Page
if selection == "Home":
  st.title("Welcome to the Real Estate Dashboard")
  st.write("This dashboard provides insights, analysis, and predictions for real estate properties in Gurgaon.")
  st.markdown("## Features:")
  st.markdown("- **Price Prediction:** Predict property prices based on features.")
  st.markdown("- **Analytical Module:**  Gain deeper understanding through data visualization.")
  st.markdown("- **Insight Model:** Explore trends and insights in the real estate market.")
  st.markdown("- **Recommender System:** Get property recommendations based on preferences.")
# Price Prediction Page
if selection == "Price Prediction":
    st.title("Price Prediction 📈")
    st.write("Enter property details to predict the estimated price range.")
    st.header('Enter your inputs:')
    # Inputs for price prediction
    property_type = st.selectbox('Property Type', ['flat', 'house'])
    sector = st.selectbox('Sector', df['sector'].unique().tolist())
    bedroom = float(st.selectbox('Bedroom', df['bedroom'].unique().tolist()))
    bathroom = float(st.selectbox('Bathroom', df['bathroom'].unique().tolist()))
    balcony = st.selectbox('Balcony', df['balcony'].unique().tolist())
    agepossession = st.selectbox('Age Possession', df['agepossession'].unique().tolist())
    built_up_area = float(st.number_input('Built Up Area'))
    servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    store_room = float(st.selectbox('Store Room', [0.0, 1.0]))
    furnish_type = st.selectbox('Furnish Type', sorted(df['furnish_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))
    # Prediction button
    if st.button('Predict'):
        data = [[property_type, sector, bedroom, bathroom, balcony,
                 agepossession, built_up_area, servant_room, store_room,
                 furnish_type, luxury_category, floor_category]]
        columns = ['property_type', 'sector', 'bedroom', 'bathroom', 'balcony',
                   'agepossession', 'built_up_area', 'servant_room', 'store_room',
                   'furnish_type', 'luxury_category', 'floor_category']
        one_df = pd.DataFrame(data, columns=columns)
        st.dataframe(one_df)
        base_price = np.expm1(pipeline.predict(one_df))[0]
        low = base_price - 0.13
        high = base_price + 0.13
        st.write(f"The price of the property is between {low:.2f} and {high:.2f}")

# Analytical Module Page
elif selection == "Analytical Module":
    st.title("Analytical Module 🔍")
    st.write('''visualizes various real estate trends and property data for Gurgaon,including scatter plots, 
    word clouds,bar charts, and box plots. It allows users to analyze data based on property type, sector, age, and other features, 
     insights into pricing, distribution, and property characteristics through interactive visualizations.''')
    st.title("Analytical Module")
    df = pd.read_csv('data/gurgaon_property_data_after_Missing_Value_Imputation (1)')
    st.markdown("### Mapping the Real Estate Trends of Gurgaon by Sector")
    new_df=pd.read_csv('data/sector wise price distribution')
    new_df=new_df.set_index('sector')
    # Scatter Plot on Mapbox
    fig_new = px.scatter_mapbox(new_df,
                            lat="lat",
                            lon="long",
                            color='Price_per_sqrt',
                            size='Built_Up_area',
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            hover_name=new_df.index,
                            zoom=10,
                            height=600,
                            width=1200)
    # Set Mapbox layout (use open-street-map if no Mapbox token is provided)
    fig_new.update_layout(mapbox_style="open-street-map")
    # Display in Streamlit
    st.plotly_chart(fig_new, use_container_width=True)
    st.markdown("### Word Cloud of Amenities in Gurgaon Real Estate Properties")
    # Load data
    df2 = pd.read_csv('data/gurgaon_properties_cleaned_data (1) (2)')
    # Ensure 'features' column contains lists
    df2['features'] = df2['features'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and "[" in x else [])
    # Generate word cloud text
    text = " ".join(feature for sublist in df2['features'] for feature in sublist)
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=800, background_color="white",stopwords=set(['s'])).generate(text)
    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


    st.markdown("### Built Up Area vs. Price Analysis by Sector")
    property_type=st.selectbox('Select Property Type',['flat','house'])
    if property_type=='house':

      fig_property_type = px.scatter(df[df['Property_type']=='house'], x='Built_Up_area', y='price',
                 title="Built Up Area vs. Price",
                 labels={'Built_Up_area': 'Built Up Area', 'price': 'Price'},
                 color='sector',  # Color the points based on the 'sector' column
                 hover_name='bedRoom',
                 color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])  # Set colors for sectors
      fig_property_type.update_layout(xaxis_range=[0, 10000], yaxis_range=[0, 25])
      st.plotly_chart(fig_property_type)

    else:

      fig_property_type = px.scatter(df[df['Property_type']=='flat'], x='Built_Up_area', y='price',
                 title="Built Up Area vs. Price",
                 labels={'Built_Up_area': 'Built Up Area', 'price': 'Price'},
                 color='sector',  # Color the points based on the 'sector' column
                 hover_name='bedRoom',
                 color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])  # Set colors for sectors
      fig_property_type.update_layout(xaxis_range=[0, 10000],  yaxis_range=[0, 25])
      # Show the plot
      st.plotly_chart(fig_property_type)
    st.markdown("### AgePossession vs. Price Analysis")
    age_price_df = df.groupby('agePossession')['price'].mean().reset_index()
    # Plot a bar chart comparing property age with average price using Plotly Express
    fig1= px.bar(age_price_df, x='agePossession', y='price',
             title="agePossession vs. Average Price",
             labels={'agePossession': 'Property Age', 'price': 'Average Price'},
             color='agePossession',
             color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
    ## show graph
    st.plotly_chart(fig1)

    st.markdown("### Bedroom Composition in Gurgaon Real Estate Market")
    sector_option=df['sector'].unique().tolist()
    sector_option.insert(0,'Overall')
    select_sector=st.selectbox('Select Sector',sector_option)
    if select_sector=='Overall':
      fig= px.pie(df, names='bedRoom')
      fig.update_traces(textinfo='percent+label',  # Show percentage and label
                    marker=dict(colors=['#FF6347', '#4CAF50', '#1E90FF']))  # Customize colors
      fig.update_layout(title='Distribution of Bedrooms Overall',  # Custom title
                    showlegend=True,  # Show legend
                    font=dict(size=14, color='black'),  # Customize font
                    margin=dict(t=40, b=40, l=40, r=40))  # Adjust margins
      st.plotly_chart(fig,use_container_width=True)
    else:

      fig= px.pie(df[df['sector']==select_sector], names='bedRoom')
      fig.update_traces(textinfo='percent+label',  # Show percentage and label
                    marker=dict(colors=['#FF6347', '#4CAF50', '#1E90FF']))  # Customize colors
      fig.update_layout(title='Distribution of Bedrooms in specific sector',  # Custom title
                    showlegend=True,  # Show legend
                    font=dict(size=14, color='black'),  # Customize font
                    margin=dict(t=40, b=40, l=40, r=40))  # Adjust margins
      st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Price Comparison of Different Bedroom Categories in Gurgaon")
    fig = px.box(df[df['bedRoom']<=4], x='bedRoom', y='price',title='Bedroom Prices in Gurgaon Real Estate')
    # Customizing the box plot
    fig.update_traces(marker=dict(color='#1E90FF'),  # Change the color of the box plots
                      boxmean='sd',  # Show mean with standard deviation
                      jitter=0.05)  # Add slight jitter to points for better visualization

    # Customize layout for a more polished look
    fig.update_layout(
        title='Distribution of Bedroom Prices in Gurgaon',  # Custom title
        title_x=0.5,  # Center align the title
        xaxis_title='Number of Bedrooms',  # X-axis title
        yaxis_title='Price in cr',  # Y-axis title
        font=dict(family='Arial', size=14, color='black'),  # Font customization
        showlegend=False,  # Hide the legend (not necessary for this plot)
        plot_bgcolor='rgba(240, 240, 240, 0.95)',  # Change the plot background color
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust plot margins
    )
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Side by side histplot of property type")
    # Create the histogram with KDE for the 'price' column where 'Property_type' is 'flat'
    fig_sns=plt.figure(figsize=(10,4))
    sns.histplot(df[df['Property_type'] == 'flat']['price'], bins=30,kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4))
    sns.histplot(df[df['Property_type'] == 'house']['price'], bins=30,kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4))
    # Customize the plot
    plt.title('Price Distribution for Flats/house in Gurgaon', fontsize=16, fontweight='bold')  # Title with font customization
    plt.xlabel('Price (in INR)', fontsize=14)  # X-axis label with fontsize
    plt.ylabel('Frequency', fontsize=14)  # Y-axis label with fontsize
    plt.grid(True)  # Enable grid lines
    plt.legend()
    plt.xlim([0,35])
    plt.tick_params(axis='both', which='major', labelsize=12)  # Customize tick labels size
    st.pyplot(fig_sns)
    st.markdown("### Age Analysis of Properties by Floor Height and Area Categories.")
    df3=pd.read_csv('data/gurgaon_property_real_estate_data_before_ordinal_encoding')
    df3['area_category'] = pd.cut(df3['Built_Up_area'], bins=[0, 2000, 8000, 35000], labels=['Small', 'Medium', 'Large'])
    # Violin plot with 'area_category' as hue
    fig_plt=plt.figure(figsize=(12,6))
    sns.violinplot(x='floor_category', y='agePossession', hue='area_category', data=df3, split=True)
    plt.title('Age of Possession by Floor Category and Build-up Area')
    plt.xlabel('Floor Category')
    plt.ylabel('Age of Possession')
    st.pyplot(fig_plt)
    st.markdown("### Pairwise Scatter Matrix of Property Features.")
    fig = px.scatter_matrix(df3[['price', 'Built_Up_area', 'bedRoom', 'bathroom','furnish_type']],
                        dimensions=['price', 'Built_Up_area', 'bedRoom', 'bathroom'],
                        title="Pairwise Scatter Matrix of Property Features",
                        color='furnish_type',  # Color by 'furnish_type'
                        color_continuous_scale='viridis',  # Custom color scale
                        opacity=0.7)  # Adjust opacity for better visualization
    fig.update_layout(width=1200,height=1200)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### How Property Prices Vary by Type.")
    fig_box = px.box(df3,
                x='Property_type',
                y='price',
                color='Property_type',  # Different colors for each category
                title="Price Distribution by Property Type",
                hover_data=['sector', 'Built_Up_area'])  # Show extra details on hover

    fig_box.update_layout(xaxis_title="Property Type",
                        yaxis_title="Price",
                        boxmode='group',  # Grouped box plots for better comparison
                        template='plotly_dark')  # Dark mode for better visualization

    st.plotly_chart(fig_box, use_container_width=True)

# Recommender System Page
elif selection == "Recommender System":
    st.title("Recommender System 👍")
    st.write("This section provides personalized property recommendations based on multiple factors,\n"
       "including facilities, price similarity, and nearby locations.")
    st.title("Recommender System")
    # Implement property recommender system here
    location_df=pickle.load(open('model/location_df (2)','rb'))
    cos_sim1=pickle.load(open('model/cosin_sim_facility','rb'))
    cos_sim2=pickle.load(open('model/cosin_sim_prices','rb'))
    cos_sim3=pickle.load(open('model/cosin_sim_location_nearby','rb'))
    def  recommender_final(property_name,n=246):
      cosine_sim_final=cos_sim1*0.5+cos_sim2*0.8+cos_sim3*1
      sim_score=list(enumerate(cosine_sim_final[location_df.index.get_loc(property_name)]))
      sim_score_sort=sorted(sim_score,key=lambda x:x[1],reverse=True)
      top_indies=[i[0]  for i in sim_score_sort[1:246]]
      top_score=[i[1]  for i in sim_score_sort[1:246]]
      top_properties=location_df.index[top_indies].tolist()
      recommender_df_loc=pd.DataFrame({'Property_type':top_properties,'similarity_score':top_score})
      return recommender_df_loc
    # st.dataframe(location_df)
    st.title('Select Location and Radius')
    selected_location=st.selectbox('Location',sorted(location_df.columns.to_list()))
    radius=st.number_input('Radius in Kms')
    if st.button('search'):
      radius_search=location_df[location_df[selected_location]<(radius*1000)][selected_location].sort_values()
      for key,values in radius_search.items():
        st.text(str(key) +" "+str(round(values/1000)) + "KMs")
    st.title('Recommend Appartments')
    selected_appartment=st.selectbox('Select Appartment',sorted(location_df.index.to_list()))
    if st.button('Recommend'):
      recommended_df=recommender_final(selected_appartment)
      st.dataframe(recommended_df)


elif selection == "Insight Model":
    st.title("Insight Model 📊")
    st.write("This page provides insights into the data and model performance of the real estate price prediction system.")

    st.markdown("### 🔍 Feature Importance (Random Forest)")
    # Access feature importances
    importances = pipeline.named_steps['regressor'].feature_importances_
    # Get feature names after transformation
    features = pipeline.named_steps['preprocessor'].get_feature_names_out()

    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Plot Feature Importance with Plotly
    fig_feat = px.bar(feature_df.head(15),
                      x='Importance',
                      y='Feature',
                      orientation='h',
                      title='Top 15 Feature Importances',
                      color='Importance',
                      color_continuous_scale='viridis')
    fig_feat.update_layout(yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_feat, use_container_width=True)
    # predicted = np.expm1(pipeline.predict(df))
    # actual = np.expm1(target)  # Replace with appropriate column
    predicted = np.expm1(pipeline.predict(df))
    actual = np.expm1(target)  # Replace with appropriate column
    # Create a DataFrame for plotting
    pred_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    
    # Plot using Plotly Express
    fig_actual_vs_pred = px.scatter(pred_df, x='Actual', y='Predicted',
                                title='Actual vs. Predicted Prices',
                                labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
                                trendline="ols")
    fig_actual_vs_pred.update_layout(width=800, height=600)
    st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

    
    residuals = actual - predicted
    # Histogram of residuals
    fig_resid_hist, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='skyblue')
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    st.pyplot(fig_resid_hist)
    
    # Q-Q plot using statsmodels
    fig_qq = sm.qqplot(residuals, line='45')
    plt.title("Q-Q Plot of Residuals")
    st.pyplot(plt.gcf())
    # Calculate regression metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Score'],
        'Value': [mae, mse, rmse, r2]
    })
    
    # Create a bar plot using Plotly Express
    fig_metrics = px.bar(
        metrics_df, 
        x='Metric', 
        y='Value',
        title="Regression Metrics",
        text='Value',
        color='Metric',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Optional layout customization for a polished look
    fig_metrics.update_layout(
        xaxis_title="Metric",
        yaxis_title="Value",
        template='plotly_white',
        showlegend=False
    )
    fig_metrics.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    # Display the plot in Streamlit
    st.plotly_chart(fig_metrics, use_container_width=True)



                                         
