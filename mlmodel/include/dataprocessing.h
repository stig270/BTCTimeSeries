#ifndef DATAPROCESSING_H
#define DATAPROCESSING_H

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>


struct DataPoint {
    std::string timestamp;
    std::vector<float> features;
    float label;
};

class TimeSeriesDataset {
private:
    std::vector<DataPoint> data;
    std::map<std::string, size_t> timestamp_to_index;
    
public:
    TimeSeriesDataset(const std::string& filename) {
        loadCSV(filename);
        buildTimestampIndex();
    }
    
    void loadCSV(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            DataPoint point;
            
            std::getline(ss, point.timestamp, ',');
            std::getline(ss, cell, ',');
            std::vector<std::string> cells;
            while (std::getline(ss, cell, ',')) {
                cells.push_back(cell);
            }
            
            for (size_t i = 0; i < cells.size() - 1; ++i) {
                point.features.push_back(std::stof(cells[i]));
            }
            point.label = std::stof(cells.back());
            
            data.push_back(point);
        }
    }
    
    void buildTimestampIndex() {
        for (size_t i = 0; i < data.size(); ++i) {
            timestamp_to_index[data[i].timestamp] = i;
        }
    }
    
    std::vector<std::string> getValidStartTimestamps(int window_size = 31) {
        std::vector<std::string> valid_timestamps;
        
        for (size_t i = 0; i < data.size() - window_size; ++i) {
            valid_timestamps.push_back(data[i].timestamp);
        }
        
        return valid_timestamps;
    }
    
    std::vector<std::vector<float>> getWindow(const std::string& start_timestamp, 
                                               int window_size = 31) {
        std::vector<std::vector<float>> window;
        
        auto it = timestamp_to_index.find(start_timestamp);
        
        
        size_t start_idx = it->second;
    
        
        for (size_t i = start_idx; i < start_idx + window_size; ++i) {
            window.push_back(data[i].features);
        }
        
        return window;
    }
    
    float getLabel(const std::string& start_timestamp, int window_size = 31) {
        auto it = timestamp_to_index.find(start_timestamp);
        size_t start_idx = it->second;
        
        return data[start_idx + window_size - 1].label;
    }
    
    size_t size() const { return data.size(); }
};

class DatasetSplitter {
public:
    std::vector<std::string> train_timestamps;
    std::vector<std::string> val_timestamps;
    
    DatasetSplitter(TimeSeriesDataset& dataset, 
                    float train_ratio = 0.8, 
                    int window_size = 31) {
        auto all_timestamps = dataset.getValidStartTimestamps(window_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_timestamps.begin(), all_timestamps.end(), gen);
        
        size_t train_size = static_cast<size_t>(all_timestamps.size() * train_ratio);
        
        train_timestamps.assign(all_timestamps.begin(), 
                               all_timestamps.begin() + train_size);
        val_timestamps.assign(all_timestamps.begin() + train_size, 
                             all_timestamps.end());
    }
};

#endif