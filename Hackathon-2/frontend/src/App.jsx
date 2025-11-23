import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [latitude, setLatitude] = useState('43.80');
  const [longitude, setLongitude] = useState('-79.70');
  const [date, setDate] = useState('2024-06-15');
  const [cropType, setCropType] = useState('maize');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log('Sending request to backend...');
      
      const response = await axios.post('http://localhost:8000/predict', {
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        date: date,
        crop_type: cropType
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('Response received:', response.data);

      if (response.data) {
        setResult(response.data);
      } else {
        setError('No data received from server');
      }
    } catch (err) {
      console.error('Error details:', err);
      
      if (err.response) {
        console.error('Response error:', err.response.status, err.response.data);
        setError(`Server error: ${err.response.status} - ${JSON.stringify(err.response.data)}`);
      } else if (err.request) {
        console.error('Request error:', err.request);
        setError('No response from server. Is the backend running on port 8000?');
      } else {
        console.error('Error message:', err.message);
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>🌾 Crop Yield Prediction System</h1>
        <p>Predict crop yields using weather and satellite data</p>
      </div>

      <div style={styles.content}>
        {/* Form Section */}
        <div style={styles.formSection}>
          <h2>📍 Location & Crop Information</h2>
          
          <div style={styles.formGroup}>
            <label style={styles.label}>Latitude</label>
            <input
              type="number"
              value={latitude}
              onChange={(e) => setLatitude(e.target.value)}
              placeholder="e.g., 43.80"
              step="0.01"
              style={styles.input}
            />
            <small style={styles.hint}>Range: -90 to 90</small>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Longitude</label>
            <input
              type="number"
              value={longitude}
              onChange={(e) => setLongitude(e.target.value)}
              placeholder="e.g., -79.70"
              step="0.01"
              style={styles.input}
            />
            <small style={styles.hint}>Range: -180 to 180</small>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Prediction Date</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              style={styles.input}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Crop Type</label>
            <select
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
              style={styles.select}
            >
              <option value="maize">Maize</option>
              <option value="wheat">Wheat</option>
              <option value="rice">Rice</option>
              <option value="soybean">Soybean</option>
              <option value="potato">Potato</option>
              <option value="sugarcane">Sugarcane</option>
            </select>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            style={{
              ...styles.button,
              opacity: loading ? 0.6 : 1,
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? '⏳ Fetching data & Predicting...' : '🚀 Get Prediction'}
          </button>
        </div>

        {/* Results Section */}
        <div style={styles.resultsSection}>
          {error && (
            <div style={styles.errorCard}>
              <h3>❌ Error</h3>
              <p style={styles.errorText}>{error}</p>
            </div>
          )}

          {loading && (
            <div style={styles.loadingCard}>
              <h3>⏳ Loading...</h3>
              <p>Fetching weather data, satellite imagery, and running predictions...</p>
              <div style={styles.loadingBar}></div>
            </div>
          )}

          {result && !error && (
            <div>
              {/* Main Prediction Card */}
              <div style={styles.resultCard}>
                <h2 style={{ color: '#2d5016', marginBottom: '20px' }}>
                  📊 Yield Prediction
                </h2>

                <div style={styles.yieldDisplay}>
                  <div style={styles.yieldValue}>
                    {Math.round(result.yield_estimate)}
                  </div>
                  <div style={styles.yieldUnit}>kg/ha</div>
                </div>

                <div style={styles.confidenceContainer}>
                  <div style={styles.confidenceLabel}>
                    Confidence Score: {Math.round(result.confidence_score * 100)}%
                  </div>
                  <div style={styles.progressBar}>
                    <div
                      style={{
                        ...styles.progressFill,
                        width: `${result.confidence_score * 100}%`
                      }}
                    />
                  </div>
                </div>

                <div style={styles.infoGrid}>
                  <div style={styles.infoItem}>
                    <span style={styles.infoLabel}>Crop Type:</span>
                    <span style={styles.infoValue}>{result.crop_type}</span>
                  </div>
                  <div style={styles.infoItem}>
                    <span style={styles.infoLabel}>Location:</span>
                    <span style={styles.infoValue}>
                      {result.location.latitude}, {result.location.longitude}
                    </span>
                  </div>
                  <div style={styles.infoItem}>
                    <span style={styles.infoLabel}>Date:</span>
                    <span style={styles.infoValue}>{result.date}</span>
                  </div>
                  <div style={styles.infoItem}>
                    <span style={styles.infoLabel}>Model RMSE:</span>
                    <span style={styles.infoValue}>{result.model_info.rmse}</span>
                  </div>
                </div>
              </div>

              {/* Weather Data Card */}
              {result.weather_data && (
                <div style={styles.resultCard}>
                  <h3 style={{ color: '#2d5016', marginBottom: '15px' }}>
                    🌤️ Weather Data (Auto-Fetched)
                  </h3>
                  <div style={styles.dataGrid}>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Temperature</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.temperature_celsius}°C
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Precipitation</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.precipitation_mm} mm
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Humidity</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.humidity_percent}%
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Wind Speed</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.wind_speed_ms} m/s
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Cloud Coverage</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.cloud_coverage_percent}%
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Description</span>
                      <span style={styles.dataValue}>
                        {result.weather_data.description}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Satellite Data Card */}
              {result.satellite_data && (
                <div style={styles.resultCard}>
                  <h3 style={{ color: '#2d5016', marginBottom: '15px' }}>
                    🛰️ Satellite Data (Auto-Fetched)
                  </h3>
                  <div style={styles.dataGrid}>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>NDVI (Vegetation Index)</span>
                      <span style={styles.dataValue}>
                        {result.satellite_data.ndvi}
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Soil Moisture</span>
                      <span style={styles.dataValue}>
                        {result.satellite_data.soil_moisture}
                      </span>
                    </div>
                    <div style={styles.dataItem}>
                      <span style={styles.dataLabel}>Source</span>
                      <span style={styles.dataValue}>
                        {result.satellite_data.source}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
    fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
  },
  header: {
    backgroundColor: '#2d5016',
    color: 'white',
    padding: '30px',
    textAlign: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  content: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '30px',
    padding: '30px',
    maxWidth: '1400px',
    margin: '0 auto'
  },
  formSection: {
    backgroundColor: 'white',
    padding: '25px',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    height: 'fit-content'
  },
  resultsSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px'
  },
  formGroup: {
    marginBottom: '20px'
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    fontWeight: 'bold',
    color: '#2d5016'
  },
  input: {
    width: '100%',
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    boxSizing: 'border-box'
  },
  select: {
    width: '100%',
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    boxSizing: 'border-box',
    backgroundColor: 'white'
  },
  hint: {
    display: 'block',
    marginTop: '4px',
    color: '#999',
    fontSize: '12px'
  },
  button: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#52a552',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '16px',
    fontWeight: 'bold',
    cursor: 'pointer'
  },
  resultCard: {
    backgroundColor: 'white',
    padding: '25px',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  },
  errorCard: {
    backgroundColor: '#fee',
    padding: '25px',
    borderRadius: '8px',
    border: '2px solid #f99',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  },
  errorText: {
    color: '#c33',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word'
  },
  loadingCard: {
    backgroundColor: '#f0f8ff',
    padding: '25px',
    borderRadius: '8px',
    border: '2px solid #87ceeb',
    textAlign: 'center'
  },
  loadingBar: {
    height: '4px',
    backgroundColor: '#87ceeb',
    borderRadius: '2px',
    marginTop: '15px',
    animation: 'pulse 1.5s ease-in-out infinite'
  },
  yieldDisplay: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '10px',
    marginBottom: '20px'
  },
  yieldValue: {
    fontSize: '48px',
    fontWeight: 'bold',
    color: '#52a552'
  },
  yieldUnit: {
    fontSize: '18px',
    color: '#666',
    fontWeight: 'bold'
  },
  confidenceContainer: {
    marginBottom: '20px'
  },
  confidenceLabel: {
    marginBottom: '8px',
    fontWeight: 'bold',
    color: '#333'
  },
  progressBar: {
    height: '10px',
    backgroundColor: '#eee',
    borderRadius: '5px',
    overflow: 'hidden'
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#52a552',
    transition: 'width 0.3s ease'
  },
  infoGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '15px',
    marginTop: '20px',
    paddingTop: '20px',
    borderTop: '1px solid #eee'
  },
  infoItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px'
  },
  infoLabel: {
    fontSize: '12px',
    color: '#999',
    fontWeight: 'bold'
  },
  infoValue: {
    fontSize: '14px',
    color: '#333',
    fontWeight: '500'
  },
  dataGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '15px'
  },
  dataItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
    padding: '10px',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px'
  },
  dataLabel: {
    fontSize: '12px',
    color: '#999',
    fontWeight: 'bold'
  },
  dataValue: {
    fontSize: '14px',
    color: '#2d5016',
    fontWeight: 'bold'
  }
};

export default App;