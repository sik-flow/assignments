
# Module 2 Assessment

Welcome to your Mod 2 Assessment. You will be tested for your understanding of concepts and ability to solve problems that have been covered in class and in the curriculum.

Use any libraries you want to solve the problems in the assessment.

The sections of the assessment are:

- Accessing Data Through APIs
- Object Oriented Programming
- SQL and Relational Databases

In this assessment you will primarily be exploring a Pokemon dataset. Pokemon are fictional creatures from the [Nintendo franchise](https://en.wikipedia.org/wiki/Pok%C3%A9mon) of the same name.

Some Pokemon facts that might be useful:
* The word "pokemon" is both singular and plural. You may refer to "one pokemon" or "many pokemon".
* Pokemon have attributes such as a name, weight, and height.
* Pokemon have one or multiple "types". A type is something like "electric", "water", "ghost", or "normal" that indicates the abilities that pokemon may possess.
* The humans who collect pokemon are called "trainers".


```python
# import the necessary libraries
import requests
import json
import pandas as pd
import sqlite3
```

## Part 1: Accessing Data Through APIs [Suggested Time: 25 minutes]

In this section we'll be using PokeAPI to get data on Pokemon. 

[Consult the documentation here](https://pokeapi.co/docs/v2.html) for information on obtaining data from this API.

### 1. Get the "types"
We want to know the "types" of any particular pokemon given it's name. Complete the `get_pokemon_types` function below. It should return a `list` of all the names of the "types" that pokemon has

Make a request to `"https://pokeapi.co/api/v2/pokemon/<add-name-of-pokemon-here>"`. Inspect the API response and extract the names of the types. Here are the [docs for this specific API route](https://pokeapi.co/docs/v2.html/#pokemon).

```python
# Examples: 
get_pokemon_types("pikachu")   # returns ["electric"]
get_pokemon_types("bulbasaur") # returns ["poison", "grass"]
get_pokemon_types("snorlax")   # returns ["normal"]
get_pokemon_types("moltres")   # returns ["flying", "fire"]
```


```python
def get_pokemon_types(name):
    '''
    input: name - a string of the pokemon's name
    
    return: a list of strings of the one or more types belonging to the pokemon
    '''
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"
    poke_data = requests.get(url).json()
    
    return [type_data["type"]["name"] for type_data in poke_data["types"]]
```


```python
get_pokemon_types("bulbasaur")
```




    ['poison', 'grass']



## Part 2: Object Oriented Programming [Suggested Time: 20 minutes]

As a pokemon trainer we want to make sure our pokemon are performing at their peak. To measure this, we want to calculate a pokemon's Body Mass Index (or BMI). This is a statistic calculated using the pokemon's height and weight. 

To help with this task we we will create Pokemon objects that methods can be called on. 

You'll be working with following dictionaries to create the `Pokemon` objects


```python
# Use the following data
bulbasaur_data = {"name": 'bulbasaur', "weight": 69, "height": 7, "base_experience": 64, "types": ["grass", "poison"]}
charmander_data = {"name": 'charmander', "weight": 85, "height": 6, "base_experience": 62, "types": ["fire"]}
squirtle_data = {"name": 'squirtle', "weight": 90, "height": 5, "base_experience": 63, "types": ["water"]}
```

### 1. Creating a Class

Create a class called `Pokemon` with an `__init__` method. Every `Pokemon` instance should have the following attributes:
* `name`
* `weight`
* `height`


```python
# Create your class below with the correct syntax, including an __init__ method.
class Pokemon:
    def __init__(self, data):
        self.name = data["name"]
        self.weight = data["weight"]
        self.height = data["height"]
        
    def bmi(self):
        return (self.weight*0.1)/(self.height*0.1)**2
        
```

    
### 2. Instantiating Objects

Using the `bulbasaur_data`, `charmander_data` and `squirtle_data` variables, create the corresponding pokemon objects.


```python
bulbasaur = Pokemon(bulbasaur_data)
charmander = Pokemon(charmander_data)
squirtle = Pokemon(squirtle_data)
```


```python
# run this cell to test and check your code
# you may need to edit the attribute variable names if you named them differently!

def print_pokeinfo(pkmn):
    print('Name: ' + pkmn.name)
    print('Weight: ' + str(pkmn.weight))
    print('Height: ' + str(pkmn.height))
    print('\n')
    
print_pokeinfo(bulbasaur)
print_pokeinfo(charmander)
print_pokeinfo(squirtle)
```

    Name: bulbasaur
    Weight: 69
    Height: 7
    
    
    Name: charmander
    Weight: 85
    Height: 6
    
    
    Name: squirtle
    Weight: 90
    Height: 5
    
    


### 3. Instance Methods

Write an instance method called `bmi` within the class `Pokemon` defined above to calculate the BMI of a Pokemon. 

BMI is defined by the formula: $\frac{weight}{height^{2}}$ 

The BMI should be calculated with weight in **kilograms** and height in **meters**. 


The height and weight data of Pokemon from the API is in **decimeters** and **hectograms** respectively. Here are the conversions:

```
1 decimeter = 0.1 meters
1 hectogram = 0.1 kilograms
```


```python
# run this cell to test and check your code

# After defining a new instance method on the class, 
# you will have to rerun the code instantiating your objects

print(bulbasaur.bmi()) # 14.08
print(charmander.bmi()) # 23.61
print(squirtle.bmi()) # 36.0
```

    14.081632653061222
    23.611111111111104
    36.0


## Part 3: SQL and Relational Databases [Suggested Time: 30 minutes]

For this section, we've put the Pokemon data into SQL tables. You won't need to use your list of dictionaries or the JSON file for this section. The schema of `pokemon.db` is as follows:

<img src="data/pokemon_db.png" alt="db schema" style="width:500px;"/>

Assign your SQL queries as strings to the variables `q1`, `q2`, etc. and run the cells at the end of this section to print your results as Pandas DataFrames.

- q1: Find all the pokemon on the "pokemon" table. Display all columns.  

  
- q2: Find all the rows from the "pokemon_types" table where the type_id is 3.


- q3: Find all the rows from the "pokemon_types" table where the associated type is "water". Do so without hard-coding the id of the "water" type, using only the name.


- q4: Find the names of all pokemon that have the "psychic" type.


- q5: Find the average weight for each type. Order the results from highest weight to lowest weight. Display the type name next to the average weight.


- q6: Find the names and ids of all the pokemon that have more than 1 type.


- q7: Find the id of the type that has the most pokemon. Display type_id next to the number of pokemon having that type. 


**Important note on syntax**: use `double quotes ""` when quoting strings **within** your query and wrap the entire query in `single quotes ''`.


```python
cnx = sqlite3.connect('data/pokemon.db')
```


```python
# q1: Find all the pokemon on the "pokemon" table. Display all columns. 
q1 = 'SELECT * FROM pokemon'
pd.read_sql(q1, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>base_experience</th>
      <th>weight</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>bulbasaur</td>
      <td>64</td>
      <td>69</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ivysaur</td>
      <td>142</td>
      <td>130</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>venusaur</td>
      <td>236</td>
      <td>1000</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>charmander</td>
      <td>62</td>
      <td>85</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>charmeleon</td>
      <td>142</td>
      <td>190</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>charizard</td>
      <td>240</td>
      <td>905</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>squirtle</td>
      <td>63</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>wartortle</td>
      <td>142</td>
      <td>225</td>
      <td>10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>blastoise</td>
      <td>239</td>
      <td>855</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>caterpie</td>
      <td>39</td>
      <td>29</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>metapod</td>
      <td>72</td>
      <td>99</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>butterfree</td>
      <td>178</td>
      <td>320</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>weedle</td>
      <td>39</td>
      <td>32</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>kakuna</td>
      <td>72</td>
      <td>100</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>beedrill</td>
      <td>178</td>
      <td>295</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>pidgey</td>
      <td>50</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>pidgeotto</td>
      <td>122</td>
      <td>300</td>
      <td>11</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>pidgeot</td>
      <td>216</td>
      <td>395</td>
      <td>15</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>rattata</td>
      <td>51</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>raticate</td>
      <td>145</td>
      <td>185</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>spearow</td>
      <td>52</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>fearow</td>
      <td>155</td>
      <td>380</td>
      <td>12</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>ekans</td>
      <td>58</td>
      <td>69</td>
      <td>20</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>arbok</td>
      <td>157</td>
      <td>650</td>
      <td>35</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>pikachu</td>
      <td>112</td>
      <td>60</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>raichu</td>
      <td>218</td>
      <td>300</td>
      <td>8</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>sandshrew</td>
      <td>60</td>
      <td>120</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>sandslash</td>
      <td>158</td>
      <td>295</td>
      <td>10</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>nidoran-f</td>
      <td>55</td>
      <td>70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>nidorina</td>
      <td>128</td>
      <td>200</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>122</td>
      <td>mr-mime</td>
      <td>161</td>
      <td>545</td>
      <td>13</td>
    </tr>
    <tr>
      <th>122</th>
      <td>123</td>
      <td>scyther</td>
      <td>100</td>
      <td>560</td>
      <td>15</td>
    </tr>
    <tr>
      <th>123</th>
      <td>124</td>
      <td>jynx</td>
      <td>159</td>
      <td>406</td>
      <td>14</td>
    </tr>
    <tr>
      <th>124</th>
      <td>125</td>
      <td>electabuzz</td>
      <td>172</td>
      <td>300</td>
      <td>11</td>
    </tr>
    <tr>
      <th>125</th>
      <td>126</td>
      <td>magmar</td>
      <td>173</td>
      <td>445</td>
      <td>13</td>
    </tr>
    <tr>
      <th>126</th>
      <td>127</td>
      <td>pinsir</td>
      <td>175</td>
      <td>550</td>
      <td>15</td>
    </tr>
    <tr>
      <th>127</th>
      <td>128</td>
      <td>tauros</td>
      <td>172</td>
      <td>884</td>
      <td>14</td>
    </tr>
    <tr>
      <th>128</th>
      <td>129</td>
      <td>magikarp</td>
      <td>40</td>
      <td>100</td>
      <td>9</td>
    </tr>
    <tr>
      <th>129</th>
      <td>130</td>
      <td>gyarados</td>
      <td>189</td>
      <td>2350</td>
      <td>65</td>
    </tr>
    <tr>
      <th>130</th>
      <td>131</td>
      <td>lapras</td>
      <td>187</td>
      <td>2200</td>
      <td>25</td>
    </tr>
    <tr>
      <th>131</th>
      <td>132</td>
      <td>ditto</td>
      <td>101</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>132</th>
      <td>133</td>
      <td>eevee</td>
      <td>65</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>133</th>
      <td>134</td>
      <td>vaporeon</td>
      <td>184</td>
      <td>290</td>
      <td>10</td>
    </tr>
    <tr>
      <th>134</th>
      <td>135</td>
      <td>jolteon</td>
      <td>184</td>
      <td>245</td>
      <td>8</td>
    </tr>
    <tr>
      <th>135</th>
      <td>136</td>
      <td>flareon</td>
      <td>184</td>
      <td>250</td>
      <td>9</td>
    </tr>
    <tr>
      <th>136</th>
      <td>137</td>
      <td>porygon</td>
      <td>79</td>
      <td>365</td>
      <td>8</td>
    </tr>
    <tr>
      <th>137</th>
      <td>138</td>
      <td>omanyte</td>
      <td>71</td>
      <td>75</td>
      <td>4</td>
    </tr>
    <tr>
      <th>138</th>
      <td>139</td>
      <td>omastar</td>
      <td>173</td>
      <td>350</td>
      <td>10</td>
    </tr>
    <tr>
      <th>139</th>
      <td>140</td>
      <td>kabuto</td>
      <td>71</td>
      <td>115</td>
      <td>5</td>
    </tr>
    <tr>
      <th>140</th>
      <td>141</td>
      <td>kabutops</td>
      <td>173</td>
      <td>405</td>
      <td>13</td>
    </tr>
    <tr>
      <th>141</th>
      <td>142</td>
      <td>aerodactyl</td>
      <td>180</td>
      <td>590</td>
      <td>18</td>
    </tr>
    <tr>
      <th>142</th>
      <td>143</td>
      <td>snorlax</td>
      <td>189</td>
      <td>4600</td>
      <td>21</td>
    </tr>
    <tr>
      <th>143</th>
      <td>144</td>
      <td>articuno</td>
      <td>261</td>
      <td>554</td>
      <td>17</td>
    </tr>
    <tr>
      <th>144</th>
      <td>145</td>
      <td>zapdos</td>
      <td>261</td>
      <td>526</td>
      <td>16</td>
    </tr>
    <tr>
      <th>145</th>
      <td>146</td>
      <td>moltres</td>
      <td>261</td>
      <td>600</td>
      <td>20</td>
    </tr>
    <tr>
      <th>146</th>
      <td>147</td>
      <td>dratini</td>
      <td>60</td>
      <td>33</td>
      <td>18</td>
    </tr>
    <tr>
      <th>147</th>
      <td>148</td>
      <td>dragonair</td>
      <td>147</td>
      <td>165</td>
      <td>40</td>
    </tr>
    <tr>
      <th>148</th>
      <td>149</td>
      <td>dragonite</td>
      <td>270</td>
      <td>2100</td>
      <td>22</td>
    </tr>
    <tr>
      <th>149</th>
      <td>150</td>
      <td>mewtwo</td>
      <td>306</td>
      <td>1220</td>
      <td>20</td>
    </tr>
    <tr>
      <th>150</th>
      <td>151</td>
      <td>mew</td>
      <td>270</td>
      <td>40</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>151 rows × 5 columns</p>
</div>




```python
# q2: Find all the rows from the "pokemon_types" table where the type_id is 3.
q2 = 'SELECT * FROM pokemon_types WHERE type_id = 3'
pd.read_sql(q2, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>pokemon_id</th>
      <th>type_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>17</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35</td>
      <td>22</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59</td>
      <td>41</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>61</td>
      <td>42</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>123</td>
      <td>83</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>125</td>
      <td>84</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>127</td>
      <td>85</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>178</td>
      <td>123</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>187</td>
      <td>130</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>205</td>
      <td>142</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>208</td>
      <td>144</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>210</td>
      <td>145</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>212</td>
      <td>146</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>216</td>
      <td>149</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# q3: Find all the rows from the "pokemon_types" table where the associated type is "water". Do so without hard-coding the id of the "water" type, using only the name.
q3 = 'SELECT pokemon_types.* FROM pokemon_types INNER JOIN types ON types.id = pokemon_types.type_id WHERE types.name = "water"'
pd.read_sql(q3, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>pokemon_id</th>
      <th>type_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>7</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>8</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>9</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>54</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>81</td>
      <td>55</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>86</td>
      <td>60</td>
      <td>11</td>
    </tr>
    <tr>
      <th>6</th>
      <td>87</td>
      <td>61</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>88</td>
      <td>62</td>
      <td>11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>102</td>
      <td>72</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>104</td>
      <td>73</td>
      <td>11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>114</td>
      <td>79</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>116</td>
      <td>80</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>128</td>
      <td>86</td>
      <td>11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>129</td>
      <td>87</td>
      <td>11</td>
    </tr>
    <tr>
      <th>14</th>
      <td>133</td>
      <td>90</td>
      <td>11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>134</td>
      <td>91</td>
      <td>11</td>
    </tr>
    <tr>
      <th>16</th>
      <td>146</td>
      <td>98</td>
      <td>11</td>
    </tr>
    <tr>
      <th>17</th>
      <td>147</td>
      <td>99</td>
      <td>11</td>
    </tr>
    <tr>
      <th>18</th>
      <td>168</td>
      <td>116</td>
      <td>11</td>
    </tr>
    <tr>
      <th>19</th>
      <td>169</td>
      <td>117</td>
      <td>11</td>
    </tr>
    <tr>
      <th>20</th>
      <td>170</td>
      <td>118</td>
      <td>11</td>
    </tr>
    <tr>
      <th>21</th>
      <td>171</td>
      <td>119</td>
      <td>11</td>
    </tr>
    <tr>
      <th>22</th>
      <td>172</td>
      <td>120</td>
      <td>11</td>
    </tr>
    <tr>
      <th>23</th>
      <td>173</td>
      <td>121</td>
      <td>11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>185</td>
      <td>129</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>186</td>
      <td>130</td>
      <td>11</td>
    </tr>
    <tr>
      <th>26</th>
      <td>188</td>
      <td>131</td>
      <td>11</td>
    </tr>
    <tr>
      <th>27</th>
      <td>192</td>
      <td>134</td>
      <td>11</td>
    </tr>
    <tr>
      <th>28</th>
      <td>197</td>
      <td>138</td>
      <td>11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>199</td>
      <td>139</td>
      <td>11</td>
    </tr>
    <tr>
      <th>30</th>
      <td>201</td>
      <td>140</td>
      <td>11</td>
    </tr>
    <tr>
      <th>31</th>
      <td>203</td>
      <td>141</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# q4: Find the names of all pokemon that have the "psychic" type.
q4 = 'SELECT pokemon.name FROM pokemon INNER JOIN pokemon_types ON pokemon_types.pokemon_id = pokemon.id INNER JOIN types ON types.id = pokemon_types.type_id WHERE types.name = "psychic"'
pd.read_sql(q4, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kadabra</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alakazam</td>
    </tr>
    <tr>
      <th>3</th>
      <td>slowpoke</td>
    </tr>
    <tr>
      <th>4</th>
      <td>slowbro</td>
    </tr>
    <tr>
      <th>5</th>
      <td>drowzee</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hypno</td>
    </tr>
    <tr>
      <th>7</th>
      <td>exeggcute</td>
    </tr>
    <tr>
      <th>8</th>
      <td>exeggutor</td>
    </tr>
    <tr>
      <th>9</th>
      <td>starmie</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mr-mime</td>
    </tr>
    <tr>
      <th>11</th>
      <td>jynx</td>
    </tr>
    <tr>
      <th>12</th>
      <td>mewtwo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mew</td>
    </tr>
  </tbody>
</table>
</div>




```python
# q5: Find the average weight for each type. Order the results from highest weight to lowest weight. Display the type name next to the average weight.
q5 = 'SELECT AVG(weight), types.name FROM pokemon INNER JOIN pokemon_types ON pokemon_types.pokemon_id = pokemon.id INNER JOIN types ON types.id = pokemon_types.type_id GROUP by types.name ORDER BY AVG(weight) DESC'
pd.read_sql(q5, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVG(weight)</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1137.000000</td>
      <td>ice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>930.454545</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>2</th>
      <td>799.357143</td>
      <td>ground</td>
    </tr>
    <tr>
      <th>3</th>
      <td>766.000000</td>
      <td>dragon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>612.473684</td>
      <td>flying</td>
    </tr>
    <tr>
      <th>5</th>
      <td>550.071429</td>
      <td>psychic</td>
    </tr>
    <tr>
      <th>6</th>
      <td>542.500000</td>
      <td>fighting</td>
    </tr>
    <tr>
      <th>7</th>
      <td>536.750000</td>
      <td>water</td>
    </tr>
    <tr>
      <th>8</th>
      <td>500.863636</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>9</th>
      <td>480.250000</td>
      <td>fire</td>
    </tr>
    <tr>
      <th>10</th>
      <td>330.000000</td>
      <td>steel</td>
    </tr>
    <tr>
      <th>11</th>
      <td>317.888889</td>
      <td>electric</td>
    </tr>
    <tr>
      <th>12</th>
      <td>264.857143</td>
      <td>grass</td>
    </tr>
    <tr>
      <th>13</th>
      <td>239.000000</td>
      <td>fairy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>238.545455</td>
      <td>poison</td>
    </tr>
    <tr>
      <th>15</th>
      <td>229.916667</td>
      <td>bug</td>
    </tr>
    <tr>
      <th>16</th>
      <td>135.666667</td>
      <td>ghost</td>
    </tr>
  </tbody>
</table>
</div>




```python
# q6: Find the names and ids the pokemon that have more than 1 type.
q6 = 'SELECT pokemon.id, pokemon.name FROM pokemon INNER JOIN pokemon_types ON pokemon.id = pokemon_types.pokemon_id GROUP BY pokemon_id HAVING COUNT(pokemon_id) > 1'
pd.read_sql(q6, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>bulbasaur</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ivysaur</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>venusaur</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>charizard</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>butterfree</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>weedle</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14</td>
      <td>kakuna</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15</td>
      <td>beedrill</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>pidgey</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17</td>
      <td>pidgeotto</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18</td>
      <td>pidgeot</td>
    </tr>
    <tr>
      <th>11</th>
      <td>21</td>
      <td>spearow</td>
    </tr>
    <tr>
      <th>12</th>
      <td>22</td>
      <td>fearow</td>
    </tr>
    <tr>
      <th>13</th>
      <td>31</td>
      <td>nidoqueen</td>
    </tr>
    <tr>
      <th>14</th>
      <td>34</td>
      <td>nidoking</td>
    </tr>
    <tr>
      <th>15</th>
      <td>39</td>
      <td>jigglypuff</td>
    </tr>
    <tr>
      <th>16</th>
      <td>40</td>
      <td>wigglytuff</td>
    </tr>
    <tr>
      <th>17</th>
      <td>41</td>
      <td>zubat</td>
    </tr>
    <tr>
      <th>18</th>
      <td>42</td>
      <td>golbat</td>
    </tr>
    <tr>
      <th>19</th>
      <td>43</td>
      <td>oddish</td>
    </tr>
    <tr>
      <th>20</th>
      <td>44</td>
      <td>gloom</td>
    </tr>
    <tr>
      <th>21</th>
      <td>45</td>
      <td>vileplume</td>
    </tr>
    <tr>
      <th>22</th>
      <td>46</td>
      <td>paras</td>
    </tr>
    <tr>
      <th>23</th>
      <td>47</td>
      <td>parasect</td>
    </tr>
    <tr>
      <th>24</th>
      <td>48</td>
      <td>venonat</td>
    </tr>
    <tr>
      <th>25</th>
      <td>49</td>
      <td>venomoth</td>
    </tr>
    <tr>
      <th>26</th>
      <td>62</td>
      <td>poliwrath</td>
    </tr>
    <tr>
      <th>27</th>
      <td>69</td>
      <td>bellsprout</td>
    </tr>
    <tr>
      <th>28</th>
      <td>70</td>
      <td>weepinbell</td>
    </tr>
    <tr>
      <th>29</th>
      <td>71</td>
      <td>victreebel</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>81</td>
      <td>magnemite</td>
    </tr>
    <tr>
      <th>38</th>
      <td>82</td>
      <td>magneton</td>
    </tr>
    <tr>
      <th>39</th>
      <td>83</td>
      <td>farfetchd</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84</td>
      <td>doduo</td>
    </tr>
    <tr>
      <th>41</th>
      <td>85</td>
      <td>dodrio</td>
    </tr>
    <tr>
      <th>42</th>
      <td>87</td>
      <td>dewgong</td>
    </tr>
    <tr>
      <th>43</th>
      <td>91</td>
      <td>cloyster</td>
    </tr>
    <tr>
      <th>44</th>
      <td>92</td>
      <td>gastly</td>
    </tr>
    <tr>
      <th>45</th>
      <td>93</td>
      <td>haunter</td>
    </tr>
    <tr>
      <th>46</th>
      <td>94</td>
      <td>gengar</td>
    </tr>
    <tr>
      <th>47</th>
      <td>95</td>
      <td>onix</td>
    </tr>
    <tr>
      <th>48</th>
      <td>102</td>
      <td>exeggcute</td>
    </tr>
    <tr>
      <th>49</th>
      <td>103</td>
      <td>exeggutor</td>
    </tr>
    <tr>
      <th>50</th>
      <td>111</td>
      <td>rhyhorn</td>
    </tr>
    <tr>
      <th>51</th>
      <td>112</td>
      <td>rhydon</td>
    </tr>
    <tr>
      <th>52</th>
      <td>121</td>
      <td>starmie</td>
    </tr>
    <tr>
      <th>53</th>
      <td>122</td>
      <td>mr-mime</td>
    </tr>
    <tr>
      <th>54</th>
      <td>123</td>
      <td>scyther</td>
    </tr>
    <tr>
      <th>55</th>
      <td>124</td>
      <td>jynx</td>
    </tr>
    <tr>
      <th>56</th>
      <td>130</td>
      <td>gyarados</td>
    </tr>
    <tr>
      <th>57</th>
      <td>131</td>
      <td>lapras</td>
    </tr>
    <tr>
      <th>58</th>
      <td>138</td>
      <td>omanyte</td>
    </tr>
    <tr>
      <th>59</th>
      <td>139</td>
      <td>omastar</td>
    </tr>
    <tr>
      <th>60</th>
      <td>140</td>
      <td>kabuto</td>
    </tr>
    <tr>
      <th>61</th>
      <td>141</td>
      <td>kabutops</td>
    </tr>
    <tr>
      <th>62</th>
      <td>142</td>
      <td>aerodactyl</td>
    </tr>
    <tr>
      <th>63</th>
      <td>144</td>
      <td>articuno</td>
    </tr>
    <tr>
      <th>64</th>
      <td>145</td>
      <td>zapdos</td>
    </tr>
    <tr>
      <th>65</th>
      <td>146</td>
      <td>moltres</td>
    </tr>
    <tr>
      <th>66</th>
      <td>149</td>
      <td>dragonite</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 2 columns</p>
</div>




```python
# q7: Find the id of the type that has the most pokemon. Display type_id next to the number of pokemon having that type. 
q7 = 'SELECT COUNT(pokemon_id) AS num_pokemon, type_id FROM pokemon_types GROUP BY type_id ORDER BY num_pokemon DESC LIMIT 1'
pd.read_sql(q7, cnx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_pokemon</th>
      <th>type_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


