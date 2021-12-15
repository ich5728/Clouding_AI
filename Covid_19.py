import pycountry
import plotly.express as px
import pandas as pd

URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
df1 = pd.read_csv(URL_DATASET)
# print(df1.head(3))
# print(df1.tail(3))
list_countries = df1['Country'].unique().tolist()
# print(list_countries)
d_country_code = {}
for country in list_countries:
    try:
        country_data = pycountry.countries.search_fuzzy(country)
        country_code = country_data[0].alpha_3
        d_country_code.update({country: country_code})
    except:
        if(country == 'Korea, South'):
            country_data = pycountry.countries.search_fuzzy('KOR')
        print('could not add ISO 3 code for ->', country)
        d_country_code.update({country:''})
print(d_country_code)

for k, v in d_country_code.items():
    df1.loc[(df1.Country == k), 'iso_alpah'] = v

fig = px.choropleth(data_frame = df1, locations=  "iso_alpha", color = "Confirmed",
 hover_name = "Country", color_continuous_scale='RdYlGn', animation_frame="Date")
fig.show()