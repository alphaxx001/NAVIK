import pandas as pd

class IOVNBDSchemaResolver:
    @staticmethod
    def _find_col(cols_orig, keywords, required_name):
        for c_orig in cols_orig:
            c_upper = str(c_orig).strip().upper()
            if all(k.upper() in c_upper for k in keywords):
                return c_orig
        raise ValueError(f"Missing required field '{required_name}'. Available columns: {cols_orig}")

    @staticmethod
    def resolve_imu_columns(df):
        cols = list(df.columns)
        
        # Check for time
        t_col = IOVNBDSchemaResolver._find_col(cols, ['TIME'], 'timestamp')
        
        # Accelerometer
        ax = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'X'], 'accelerometer X')
        ay = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'Y'], 'accelerometer Y')
        az = IOVNBDSchemaResolver._find_col(cols, ['ACCEL', 'Z'], 'accelerometer Z')
        
        # Gyroscope
        gx = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'X'], 'gyroscope X')
        gy = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'Y'], 'gyroscope Y')
        gz = IOVNBDSchemaResolver._find_col(cols, ['GYRO', 'Z'], 'gyroscope Z')

        return {
            'time': t_col,
            'accel_x': ax,
            'accel_y': ay,
            'accel_z': az,
            'gyro_x': gx,
            'gyro_y': gy,
            'gyro_z': gz
        }
        
    @staticmethod
    def resolve_v_columns(df):
        cols = list(df.columns)
        
        # Time
        t_col = IOVNBDSchemaResolver._find_col(cols, ['TIME'], 'timestamp')
        
        # GPS
        lat = IOVNBDSchemaResolver._find_col(cols, ['LAT'], 'GPS latitude')
        lon = IOVNBDSchemaResolver._find_col(cols, ['LON'], 'GPS longitude')
        
        # Velocity and Heading
        vel = IOVNBDSchemaResolver._find_col(cols, ['VELOCITY'], 'vehicle velocity')
        try:
            head = IOVNBDSchemaResolver._find_col(cols, ['HEADING'], 'vehicle heading')
        except ValueError:
            head = None # Sometimes heading might be absent, we'll gracefully handle it later
            
        try:
            yaw_rate = IOVNBDSchemaResolver._find_col(cols, ['YAW RATE'], 'vehicle yaw rate')
        except ValueError:
            yaw_rate = None
            
        return {
            'time': t_col,
            'lat': lat,
            'lon': lon,
            'velocity': vel,
            'heading': head,
            'yaw_rate': yaw_rate
        }
