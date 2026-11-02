# EDA Findings - India Housing Prices

Dataset: 250,000 listings, 20 states, 42 cities. Prices in INR lakhs; price per sq ft shown in INR.


## 1-5  Price & Size

**Q1. Distribution of prices**  
Prices run 10-500 lakhs, mean 255, median 254, skew 0.01. The shape is almost flat (uniform) rather than the right-skew normally seen in real markets - a hint that the data is synthetic.

**Q2. Distribution of sizes**  
Sizes span 500-5,000 sq ft (mean 2,750), again roughly uniform. A 1 BHK averages the same size as a 5 BHK, which does not happen in real listings.

**Q3. Price per sq ft by property type**  
Apartment: mean INR 13,047, median INR 9,245; Independent House: mean INR 13,102, median INR 9,305; Villa: mean INR 13,026, median INR 9,190. Differences between types are under 1%, i.e. property type does not drive price here.

**Q4. Size vs price**  
Pearson r = -0.003: effectively **no relationship**. In real markets this is typically strongly positive.

**Q5. Outliers**  
IQR rule flags 19,723 listings (7.9%) as high price-per-sq-ft outliers (max INR 99,182); none for price or size, which are bounded. The PPSF outliers are small, expensive listings (avg 841 sq ft at 388 lakhs vs 2,750 sq ft at 255 lakhs overall) - flagged, not removed.


## 6-10  Location

**Q6. Avg PPSF by state**  
Highest: Karnataka (INR 13,252); lowest: Punjab (INR 12,931). Spread across states is only 2.5%.

**Q7. Avg price by city**  
Highest: Bangalore (258.5 lakhs); lowest: Cuttack (250.8 lakhs). All 42 cities sit within 3.1% of each other.

**Q8. Median age by locality**  
Across the 500 localities the median age ranges 16-21 years (overall median 18); oldest stock: Locality_72 (21 y), newest: Locality_443 (16 y). Locality labels repeat across all 42 cities, so they are analysed at label level (~500 listings each); a city-locality pair has only ~12 listings, too few for stable medians.

**Q9. BHK across cities**  
Every city has a near-even 1-5 BHK split (each ~20% +/- 0.5).

**Q10. Top-5 locality price trends**  
The data has no listing dates, so a true time trend is impossible; construction period is used as a proxy. Top 5 by average price: Locality_395 (275 L), Locality_366 (273 L), Locality_497 (270 L), Locality_360 (270 L), Locality_203 (269 L), only 8% above the overall mean. Lines move up and down with no consistent direction (each point is ~70 listings, so swings are sampling noise).


## 11-15  Relationships & Correlation

**Q11. Numeric correlations**  
Strongest pairs: Price_per_SqFt_INR vs Size_in_SqFt = -0.61; Floor_No vs Total_Floors = 0.58; Price_in_Lakhs vs Price_per_SqFt_INR = 0.56. The strong ones are mechanical: Price_per_SqFt is derived from price and size, and Floor_No was capped at Total_Floors during cleaning. Among the independent raw features (BHK, size, price, age, schools, hospitals, total floors, amenity count) the largest |r| is 0.006.

**Q12. Nearby schools vs PPSF**  
r = -0.000. Average PPSF moves by less than 3.1% from fewest to most - no meaningful link.

**Q13. Nearby hospitals vs PPSF**  
r = -0.000. Average PPSF moves by less than 2.6% from fewest to most - no meaningful link.

**Q14. Price by furnishing**  
Furnished: 254.4 lakhs; Semi-furnished: 254.3 lakhs; Unfurnished: 255.0 lakhs. Furnishing carries no price premium in this data.

**Q15. PPSF by facing**  
East: 13,024; North: 13,025; South: 13,044; West: 13,140. Spread is 0.9%.


## 16-20  Investment, Amenities & Ownership

**Q16. Properties per owner type**  
Broker: 83,479; Owner: 83,268; Builder: 83,253 (near-equal thirds).

**Q17. Properties per availability status**  
Under_Construction: 125,035; Ready_to_Move: 124,965 (about 50/50).

**Q18. Parking vs price**  
Parking=No: 254.4 lakhs; Parking=Yes: 254.7 lakhs. Difference 0.3 lakhs (<1%).

**Q19. Amenities vs PPSF**  
Clubhouse: +0.2%; Garden: +0.1%; Gym: +0.8%; Playground: -0.2%; Pool: +0.3%. No amenity moves price per sq ft by more than about +/-1%.

**Q20. Transport vs PPSF / investment potential**  
PPSF: Low 13,086; Medium 13,062; High 13,027. Good-Investment share: Low 17.1%; Medium 27.4%; High 37.7%. Price is unaffected, but better transport raises the share of Good Investments because it lifts the infrastructure score in the growth model.


## Brief-specific EDA bullets

**Area vs investment return**  
r(Size, Annual_Growth_Rate) at listing level = 0.002. Growth is driven by infrastructure, metro status, availability and age, not size.

**Crime rate vs Good Investment**  
The dataset has **no crime-rate column**, so this cannot be tested directly. Infrastructure_Score (schools, hospitals, transport) is used as the area-quality proxy instead.

**Infrastructure score vs resale value**  
r(Infrastructure_Score, Price_in_Lakhs) = -0.001 - no link to price; it only affects projected growth through the assumed model.


## Data-quality notes & caveats

- **Floor_No > Total_Floors in 116,304 rows (46.5%)** - impossible in reality. Floors were capped at Total_Floors and the original issue is kept in `Floor_Data_Flag`.
- **Price_per_SqFt in the raw file is in lakhs and rounded to 2 dp** (0.05, 0.10 ...), so it was recomputed in INR from price and size.
- **Locality names (Locality_1 ... Locality_500) repeat across all 42 cities**, so locality is only meaningful as City + Locality.
- **Zero missing values and zero duplicates.**
- **Features are statistically independent of price** (every correlation with price < 0.01), consistent with synthetic data. A model cannot learn real market structure from it. The 5-year price and the Good-Investment label are therefore built from documented rules (see `src/config.py`), and model scores measure how well the rule is recovered, not real-world forecasting skill.
