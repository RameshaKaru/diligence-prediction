## Adding new data

- Data files (Excel files in the agreed format) can be added to /data folder and referred in the config file (Config file must be present in the root package. i.e. one level above this)
- An example config file is available below

```yaml
# details of the data files
datafiles:
# Excel files in agreed format
  - name: "Master_Dataset_KB-Google.xlsx"
    location: "data/"
    start_date: 2017, 02, 09                # format: year,month,date
    end_date: 2020, 03, 17
  - name: "Google_KB_Test_Dataset.xlsx"
    location: "data/"
    start_date: 2020, 05, 01
    end_date: 2020, 08, 20
# csv file with the sub center ids, where the camps will be held in the next prediction window
# this file is needed to filter out only the needed ANMs, so that it will be easier to rank them
next_sub_center_ids_file: "data/next_sub_center_ids.csv"
# dates to be ignored. Keep blank if no dates should be ignored
# if multiple date ranges are to be ignored, add them likewise
ignore_dates:
    - start_date: 2020, 03, 18
      end_date: 2020, 04, 30
```


#### Configuration parameter description

- datafiles: Array which holds the details of all data files
- name: Name of the excel file
- location: Location of the data files from the root package. Ex: "data/"
- start_date: Starting date of the data file to be considered. Dates before this will be ignored.
    - Please provide in the format _year, month, date_
- end_date: Final date to be considered in the data file
- ignore_dates: Date ranges to be ignored
- next_sub_center_ids_file: 
    - csv file with the sub center ids, where the camps will be held in the next prediction window.
    - this file is needed to filter out only the needed ANMs, so that it will be easier to rank them
    - provide the csv file location with name from the root package
    - include the sub center ids under the column name 'sub_center_id' in the csv
 
     
 _Note: Please add atleast 1 year of data as input_  
    
> Main documentation [link](README.md)