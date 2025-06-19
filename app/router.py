import sys
import pandas as pd
import numpy as np
import requests
from geopy.distance import geodesic
from sklearn.mixture import GaussianMixture
from datetime import datetime


class SafePathRouter:
    def __init__(self, crime_data_path):
        try:
            self.crime_data = pd.read_csv(crime_data_path)
            self.crime_data = self.crime_data.dropna(subset=['Latitude', 'Longitude'])

            if 'CMPLNT_FR_TM' in self.crime_data.columns:
                self.crime_data['hour'] = self.crime_data['CMPLNT_FR_TM'].apply(
                    lambda x: int(str(x).split(':')[0]) if isinstance(x, str) and ':' in str(x) else 12
                )
                self.hourly_crimes = self.crime_data.groupby('hour').size()
                total_crimes = self.hourly_crimes.sum()
                self.hourly_crime_prob = self.hourly_crimes / total_crimes
                print(f"Processed time data for {len(self.crime_data)} crime records", file=sys.stderr)
            else:
                print("Time column not found in crime data, time-based analysis disabled", file=sys.stderr)
                self.hourly_crime_prob = None

            print(f"Loaded {len(self.crime_data)} crime records", file=sys.stderr)
        except Exception as e:
            print(f"Error loading crime data: {e}", file=sys.stderr)
            raise

    def find_safe_locations(self, location, radius=2000):
        safe_locations = []
        safe_amenities = [
            'police', 'hospital', 'pharmacy', 'fire_station',
            'library', 'university', 'school', 'cafe', 'restaurant',
            'fast_food', 'bank', 'atm', 'fuel', 'convenience', 'supermarket'
        ]

        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            (
              node["amenity"~"{'|'.join(safe_amenities)}"](around:{radius},{location[0]},{location[1]});
              way["amenity"~"{'|'.join(safe_amenities)}"](around:{radius},{location[0]},{location[1]});
              node["shop"](around:{radius},{location[0]},{location[1]});
              way["shop"](around:{radius},{location[0]},{location[1]});
            );
            out center;
            """
            response = requests.post(overpass_url, data=query)
            data = response.json()

            for element in data.get('elements', []):
                if element['type'] == 'node':
                    lat = element.get('lat')
                    lon = element.get('lon')
                elif element['type'] == 'way' and 'center' in element:
                    lat = element['center'].get('lat')
                    lon = element['center'].get('lon')
                else:
                    continue

                if not lat or not lon:
                    continue

                tags = element.get('tags', {})
                name = tags.get('name', 'Unnamed Location')
                amenity_type = tags.get('amenity', tags.get('shop', 'public_place'))
                opening_hours = tags.get('opening_hours', '')
                is_24h = '24/7' in opening_hours or '24' in opening_hours

                safety_type_rating = 0
                if amenity_type in ['police', 'fire_station', 'hospital']:
                    safety_type_rating = 10
                elif amenity_type in ['pharmacy', 'bank', 'university', 'library']:
                    safety_type_rating = 8
                elif amenity_type in ['supermarket', 'convenience', 'fuel']:
                    safety_type_rating = 7
                elif amenity_type in ['cafe', 'restaurant', 'fast_food']:
                    safety_type_rating = 6
                else:
                    safety_type_rating = 5

                if is_24h:
                    safety_type_rating += 2

                safe_location = {
                    'id': str(element.get('id', '')),
                    'name': name,
                    'coordinates': {'latitude': lat, 'longitude': lon},
                    'type': amenity_type,
                    'is_24h': is_24h,
                    'safety_type_rating': safety_type_rating,
                    'address': tags.get('addr:street', '') + ' ' + tags.get('addr:housenumber', ''),
                    'categories': [amenity_type]
                }
                safe_locations.append(safe_location)

            print(f"Found {len(safe_locations)} potential safe locations", file=sys.stderr)
            return safe_locations

        except Exception as e:
            print(f"Error fetching safe locations: {e}", file=sys.stderr)
            return []

    def analyze_crime_clusters(self, n_clusters=3):
        try:
            crime_coords = self.crime_data[['Latitude', 'Longitude']].values
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(crime_coords)
            labels = gmm.predict(crime_coords)
            cluster_centers = gmm.means_
            return cluster_centers, labels
        except Exception as e:
            print(f"Error analyzing crime clusters: {e}", file=sys.stderr)
            return np.array([[40.7128, -74.0060]]), np.zeros(len(self.crime_data))

    def calculate_safety_score(self, location, crime_centers, time_weight=None, current_hour=None):
        try:
            distances = [geodesic(location, center).kilometers for center in crime_centers]

            if time_weight and current_hour is not None and self.hourly_crime_prob is not None:
                relevant_crimes = self.crime_data[
                    (self.crime_data['hour'] >= (current_hour - 2) % 24) &
                    (self.crime_data['hour'] <= (current_hour + 2) % 24)
                ]

                if len(relevant_crimes) > 0:
                    relevant_coords = relevant_crimes[['Latitude', 'Longitude']].values
                    min_crime_distance = min([
                        geodesic(location, (lat, lon)).kilometers
                        for lat, lon in relevant_coords
                    ]) if len(relevant_coords) > 0 else 10.0
                    time_safety_factor = min(1.0, min_crime_distance / 2.0)
                else:
                    time_safety_factor = 1.0
            else:
                time_safety_factor = 0.8

            avg_distance = np.mean(distances)
            min_distance = min(distances)
            safety_score = (5.0 + (avg_distance * 2.0) + (min_distance * 3.0)) * time_safety_factor
            return min(10.0, safety_score)
        except Exception as e:
            print(f"Error calculating safety score: {e}", file=sys.stderr)
            return 5.0

    def generate_safe_route(self, start, end, current_time=None):
        try:
            if current_time is None:
                current_time = datetime.now()
            current_hour = current_time.hour
            time_weight = self._calculate_time_weight(current_hour)
            print(f"Generating route for time: {current_hour}:00, time weights calculated", file=sys.stderr)

            crime_centers, _ = self.analyze_crime_clusters()

            mid_lat = (start[0] + end[0]) / 2
            mid_lng = (start[1] + end[1]) / 2
            direct_distance = geodesic(start, end).kilometers
            search_radius = max(1000, int(direct_distance * 1000))
            midpoint = (mid_lat, mid_lng)
            safe_locations = self.find_safe_locations(midpoint, radius=search_radius)

            start_safety = self.calculate_safety_score(start, crime_centers, time_weight, current_hour)
            end_safety = self.calculate_safety_score(end, crime_centers, time_weight, current_hour)

            waypoints = []
            for loc in safe_locations:
                loc_coords = (loc['coordinates']['latitude'], loc['coordinates']['longitude'])
                crime_safety_score = self.calculate_safety_score(
                    loc_coords, crime_centers, time_weight, current_hour
                )
                combined_safety = (crime_safety_score * 0.7) + (loc['safety_type_rating'] * 0.3)
                direct_path_dist = self._point_to_line_distance(loc_coords, start, end)
                path_penalty = min(1.0, direct_path_dist / (direct_distance * 0.5))
                waypoint_score = combined_safety * (1 - path_penalty)
                waypoints.append({
                    'id': loc['id'],
                    'name': loc['name'],
                    'coordinates': loc['coordinates'],
                    'safety_score': float(combined_safety),
                    'address': loc['address'],
                    'categories': loc['categories'],
                    'is_24h': loc['is_24h'],
                    'waypoint_score': float(waypoint_score)
                })

            waypoints.sort(key=lambda x: x['waypoint_score'], reverse=True)
            top_waypoints = waypoints[:15]

            return {
                'start': {
                    'coordinates': {'latitude': start[0], 'longitude': start[1]},
                    'safety_score': float(start_safety)
                },
                'end': {
                    'coordinates': {'latitude': end[0], 'longitude': end[1]},
                    'safety_score': float(end_safety)
                },
                'waypoints': top_waypoints,
                'overall_safety_score': float(np.mean([start_safety, end_safety] + [w['safety_score'] for w in top_waypoints]))
            }

        except Exception as e:
            print(f"Error generating safe route: {e}", file=sys.stderr)
            return {'error': str(e), 'message': 'Failed to generate safe route'}

    def _point_to_line_distance(self, point, line_start, line_end):
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        l2 = (x2 - x1)**2 + (y0 - y1)**2
        if l2 == 0:
            return geodesic(point, line_start).kilometers
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))
        projection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return geodesic(point, projection).kilometers

    def _calculate_time_weight(self, current_hour):
        if self.hourly_crime_prob is None:
            return {hour: 1.0 for hour in range(24)}

        weights = {}
        for hour in range(24):
            hour_diff = min(abs(hour - current_hour), 24 - abs(hour - current_hour))
            if hour_diff <= 2:
                weights[hour] = 3.0
            elif hour_diff <= 4:
                weights[hour] = 2.0
            else:
                weights[hour] = 1.0
            try:
                hour_crime_prob = self.hourly_crime_prob.get(hour, 0)
                weights[hour] *= (1.0 + hour_crime_prob * 10)
            except:
                pass
        return weights
