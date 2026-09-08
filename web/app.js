/**
 * NAVIK — Intelligent Dead Reckoning Dashboard Controller
 * Real-time replay, sensor telemetry waveforms, and geospatial map matching
 */

class NavikDashboard {
    constructor() {
        this.data = null;
        this.currentIndex = 0;
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.manualOutage = false;
        
        // Leaflet Map & Layers
        this.map = null;
        this.vehicleMarker = null;
        this.gtPolyline = null;
        this.fusedPolyline = null;
        this.rawDrPolyline = null;
        this.roadLayers = [];

        // Waveform Buffers
        this.accelHistory = { x: [], y: [], z: [] };
        this.gyroHistory = { x: [], y: [], z: [] };
        this.maxWaveformPoints = 80;

        // Elements
        this.dom = {
            playBtn: document.getElementById('playBtn'),
            playIcon: document.getElementById('playIcon'),
            resetBtn: document.getElementById('resetBtn'),
            timelineSlider: document.getElementById('timelineSlider'),
            timeElapsed: document.getElementById('timeElapsed'),
            timeTotal: document.getElementById('timeTotal'),
            phoneModeBtn: document.getElementById('phoneModeBtn'),
            toggleOutageBtn: document.getElementById('toggleOutageBtn'),
            speedBtns: document.querySelectorAll('.btn-speed'),
            
            modeDot: document.getElementById('modeDot'),
            modeValue: document.getElementById('modeValue'),
            blackoutBanner: document.getElementById('blackoutBanner'),
            gnssStatusText: document.getElementById('gnssStatusText'),
            gnssBars: document.querySelectorAll('.gnss-signal-bar'),
            mapStatusPill: document.getElementById('mapStatusPill'),
            
            speedValue: document.getElementById('speedValue'),
            speedGaugeCircle: document.getElementById('speedGaugeCircle'),
            zuptMeterFill: document.getElementById('zuptMeterFill'),
            zuptState: document.getElementById('zuptState'),
            zuptProb: document.getElementById('zuptProb'),
            compassDial: document.getElementById('compassDial'),
            headingReadout: document.getElementById('headingReadout'),
            
            errorVal: document.getElementById('errorVal'),
            driftVal: document.getElementById('driftVal'),
            accelCanvas: document.getElementById('accelCanvas'),
            gyroCanvas: document.getElementById('gyroCanvas'),
            
            // Sensor & Modal Elements
            sensorModal: document.getElementById('sensorModal'),
            btnCloseModal: document.getElementById('btnCloseModal'),
            btnSimulateSensors: document.getElementById('btnSimulateSensors'),
            sensorLiveReadout: document.getElementById('sensorLiveReadout'),
            
            // Driver Mode Elements
            btnDriverView: document.getElementById('btnDriverView'),
            btnEngineerView: document.getElementById('btnEngineerView'),
            storyBadge: document.getElementById('storyBadge'),
            storyModeLabel: document.getElementById('storyModeLabel'),
            storyDesc: document.getElementById('storyDesc'),
            
            // Location Tracking Elements
            btnLocateMe: document.getElementById('btnLocateMe'),
            locationTagText: document.getElementById('locationTagText'),

            // GNSS Blackout Flow UI Elements (SIH 26168 Sequential 4-State UI)
            flowOverlay: document.getElementById('gnssFlowOverlay'),
            flowStepBtns: document.querySelectorAll('.btn-flow-step'),
            flowStatusPill: document.getElementById('flowStatusPill'),
            flowPillIcon: document.getElementById('flowPillIcon'),
            flowPillText: document.getElementById('flowPillText'),
            flowSlideBanner: document.getElementById('flowSlideBanner'),
            flowConfidenceBadge: document.getElementById('flowConfidenceBadge'),
            flowConfNum: document.getElementById('flowConfNum'),
            flowBottomCard: document.getElementById('flowBottomCard'),
            flowCardLabel: document.getElementById('flowCardLabel'),
            flowCardHero: document.getElementById('flowCardHero'),
            flowHeroNum: document.getElementById('flowHeroNum'),
            flowHeroUnit: document.getElementById('flowHeroUnit'),
            btnToggleDrawer: document.getElementById('btnToggleDrawer'),
            cockpitDrawer: document.getElementById('cockpitDrawer')
        };

        // Sequential 4-State Engine
        this.flowState = 1;
        this.blackoutElapsedSecs = 134; // default "02:14" for instant review
        this.currentSpeed = 42;
        this.flowTransitionTimer = null;
        this.candidateRoadPoly = null;
        this.connectorPoly = null;
        this.connectorDot = null;
        this.lastReportedMode = 'GNSS';

        this.init();
    }

    async init() {
        this.initMap();
        this.bindEvents();
        await this.loadTelemetry();
        if (this.data && this.data.frames.length > 0) {
            this.setupInitialState();
            this.renderFrame(0);
            this.isPlaying = true;
            if (this.dom.playIcon) this.dom.playIcon.innerText = '❚❚';
            this.startLoop();
        }
    }

    initMap() {
        // Initialize Leaflet Map centered on Midlands UK default
        this.map = L.map('map', {
            zoomControl: false,
            attributionControl: false
        }).setView([52.5618, -1.4552], 16);

        L.control.zoom({ position: 'topright' }).addTo(this.map);

        // 1. Esri World Dark Gray (Clean unwatermarked dark basemap - Primary Default)
        const esriDark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri',
            maxZoom: 16
        }).addTo(this.map);

        // 2. CARTO Dark Matter
        const cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        });

        // 3. OpenStreetMap Standard
        const osmStandard = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
            maxZoom: 19
        });

        // Add Layer Switcher Control
        const baseMaps = {
            "Esri Dark Canvas (Clean)": esriDark,
            "CARTO Dark Matter": cartoDark,
            "OpenStreetMap Standard": osmStandard
        };
        L.control.layers(baseMaps, null, { position: 'topright' }).addTo(this.map);

        // Ensure proper rendering bounds across mobile and window resizes
        window.addEventListener('resize', () => {
            if (this.map) this.map.invalidateSize();
        });
        [100, 300, 800].forEach(ms => {
            setTimeout(() => {
                if (this.map) this.map.invalidateSize();
            }, ms);
        });

        // Trajectory Layers
        this.gtPolyline = L.polyline([], {
            color: '#3b82f6',
            weight: 3,
            dashArray: '5, 8',
            opacity: 0.8
        }).addTo(this.map);

        this.rawDrPolyline = L.polyline([], {
            color: '#ef4444',
            weight: 2,
            dashArray: '4, 6',
            opacity: 0.5
        }).addTo(this.map);

        this.fusedPolyline = L.polyline([], {
            color: '#00f0ff',
            weight: 4,
            opacity: 0.95
        }).addTo(this.map);

        // Vehicle Marker
        const vehicleIcon = L.divIcon({
            className: 'vehicle-marker-icon',
            html: `
                <div style="position:relative; width:32px; height:32px; display:flex; align-items:center; justify-content:center;">
                    <div style="position:absolute; width:28px; height:28px; border-radius:50%; background:rgba(0,240,255,0.3); animation:pulse 1.2s infinite alternate;"></div>
                    <div id="vehicleArrow" style="width:0; height:0; border-left:7px solid transparent; border-right:7px solid transparent; border-bottom:18px solid #00f0ff; filter:drop-shadow(0 0 6px #00f0ff); transform-origin:50% 50%;"></div>
                </div>
            `,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        });

        this.vehicleMarker = L.marker([52.5618, -1.4552], { icon: vehicleIcon }).addTo(this.map);

        // Candidate road and snap connector for Flow State 3
        this.candidateRoadPoly = L.polyline([], {
            color: '#64748b',
            weight: 3,
            dashArray: '3, 5',
            opacity: 0.85
        }).addTo(this.map);

        this.connectorPoly = L.polyline([], {
            color: '#f59e0b',
            weight: 2,
            dashArray: '3, 3',
            opacity: 0.95
        }).addTo(this.map);

        this.connectorDot = L.circleMarker([0, 0], {
            radius: 4,
            color: '#f59e0b',
            fillColor: '#ffffff',
            fillOpacity: 1,
            weight: 2
        });
    }

    async loadTelemetry() {
        try {
            const resp = await fetch('data/telemetry.json');
            this.data = await resp.json();
            
            // Render road network
            if (this.data.roads) {
                this.data.roads.forEach(road => {
                    const poly = L.polyline(road.coords, {
                        color: road.type === 'primary' ? '#475569' : '#334155',
                        weight: road.type === 'primary' ? 4 : 2,
                        opacity: 0.6
                    }).addTo(this.map);
                    this.roadLayers.push(poly);
                });
            }
        } catch (e) {
            console.error('Failed to load telemetry.json', e);
        }
    }

    setupInitialState() {
        const frames = this.data.frames;
        this.dom.timelineSlider.max = frames.length - 1;
        const totalSecs = frames[frames.length - 1].t;
        this.dom.timeTotal.innerText = this.formatTime(totalSecs);

        // Center map on first frame
        const first = frames[0];
        this.map.setView(first.fused, 16);
    }

    bindEvents() {
        // Play / Pause
        this.dom.playBtn.addEventListener('click', () => {
            this.isPlaying = !this.isPlaying;
            this.dom.playIcon.innerText = this.isPlaying ? '❚❚' : '▶';
        });

        // Reset
        this.dom.resetBtn.addEventListener('click', () => {
            this.currentIndex = 0;
            this.dom.timelineSlider.value = 0;
            this.renderFrame(0);
        });

        // Scrub
        this.dom.timelineSlider.addEventListener('input', (e) => {
            this.currentIndex = parseInt(e.target.value, 10);
            this.renderFrame(this.currentIndex);
        });

        // Speed Buttons
        this.dom.speedBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.dom.speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.playbackSpeed = parseFloat(btn.dataset.speed);
            });
        });

        // Outage Toggle
        this.dom.toggleOutageBtn.addEventListener('click', () => {
            this.manualOutage = !this.manualOutage;
            this.dom.toggleOutageBtn.innerText = this.manualOutage ? 'Restore GNSS' : 'Inject Outage';
            this.dom.toggleOutageBtn.style.background = this.manualOutage ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.15)';
            this.dom.toggleOutageBtn.style.color = this.manualOutage ? '#10b981' : '#ef4444';

            if (this.manualOutage) {
                this.blackoutElapsedSecs = 0;
                this.setFlowState(2, true);
            } else {
                this.setFlowState(4, true);
            }
        });

        // Phone Live Sensor Mode Toggle
        if (this.dom.phoneModeBtn) {
            this.dom.phoneModeBtn.addEventListener('click', () => {
                this.toggleLivePhoneMode();
            });
        }

        // Sensor Modal Event Listeners
        if (this.dom.btnCloseModal) {
            this.dom.btnCloseModal.addEventListener('click', () => {
                if (this.dom.sensorModal) this.dom.sensorModal.classList.add('hidden');
            });
        }

        if (this.dom.btnSimulateSensors) {
            this.dom.btnSimulateSensors.addEventListener('click', () => {
                if (this.dom.sensorModal) this.dom.sensorModal.classList.add('hidden');
                this.startSimulatedPhoneSensors();
            });
        }
        // Fullscreen / Expand Map Toggle
        const btnExpand = document.getElementById('btnExpandMap');
        if (btnExpand) {
            btnExpand.addEventListener('click', () => {
                const mapSection = document.querySelector('.map-section');
                mapSection.classList.toggle('fullscreen-map');
                const isFull = mapSection.classList.contains('fullscreen-map');
                document.body.classList.toggle('in-fullscreen-map', isFull);
                btnExpand.innerHTML = isFull ? '✕ Exit' : '⛶ Fullscreen';
                btnExpand.classList.toggle('active', isFull);
                [50, 150, 300, 600].forEach(ms => {
                    setTimeout(() => {
                        if (this.map) this.map.invalidateSize();
                    }, ms);
                });
            });
        }

        // Floating HUD Toggle Button
        const btnToggleHud = document.getElementById('btnToggleHud');
        if (btnToggleHud) {
            btnToggleHud.addEventListener('click', () => {
                const hud = document.getElementById('mapHudOverlay');
                if (hud) {
                    hud.classList.toggle('hidden');
                    btnToggleHud.classList.toggle('active', !hud.classList.contains('hidden'));
                }
            });
        }

        // Close/Dismiss Blackout Banner Button
        const btnCloseBanner = document.getElementById('btnCloseBanner');
        if (btnCloseBanner) {
            btnCloseBanner.addEventListener('click', (e) => {
                e.stopPropagation();
                if (this.dom.blackoutBanner) {
                    this.dom.blackoutBanner.classList.add('user-dismissed');
                    this.dom.blackoutBanner.classList.add('hidden');
                }
            });
        }

        // Mode View Switcher (Driver Mode vs Engineer HUD)
        if (this.dom.btnDriverView && this.dom.btnEngineerView) {
            this.dom.btnDriverView.addEventListener('click', () => {
                this.dom.btnDriverView.classList.add('active');
                this.dom.btnEngineerView.classList.remove('active');
                document.body.classList.add('driver-mode');
                setTimeout(() => this.map && this.map.invalidateSize(), 150);
            });

            this.dom.btnEngineerView.addEventListener('click', () => {
                this.dom.btnEngineerView.classList.add('active');
                this.dom.btnDriverView.classList.remove('active');
                document.body.classList.remove('driver-mode');
                setTimeout(() => this.map && this.map.invalidateSize(), 150);
            });
        }

        // Locate Me Button
        if (this.dom.btnLocateMe) {
            this.dom.btnLocateMe.addEventListener('click', () => {
                this.locateUser();
            });
        }

        // Benchmark Route Return Button
        const btnBenchmark = document.getElementById('btnBenchmarkRoute');
        if (btnBenchmark) {
            btnBenchmark.addEventListener('click', () => {
                this.flyToBenchmark();
            });
        }

        // Floating Location Tag Click
        const locTag = document.getElementById('mapLocationTag');
        if (locTag) {
            locTag.addEventListener('click', () => {
                if (this.isLivePhoneMode) {
                    this.locateUser();
                } else {
                    this.flyToBenchmark();
                }
            });
        }

        // GNSS Blackout Flow Stepper buttons (1, 2, 3, 4)
        if (this.dom.flowStepBtns) {
            this.dom.flowStepBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const stepNum = parseInt(btn.dataset.step, 10);
                    this.setFlowState(stepNum, false);
                });
            });
        }

        // Collapsible Cockpit Controls Drawer Toggle
        if (this.dom.btnToggleDrawer && this.dom.cockpitDrawer) {
            this.dom.btnToggleDrawer.addEventListener('click', (e) => {
                e.preventDefault();
                this.dom.cockpitDrawer.classList.toggle('hidden');
                this.dom.btnToggleDrawer.classList.toggle('active');
                setTimeout(() => this.map && this.map.invalidateSize(), 200);
            });
        }
    }

    toggleLivePhoneMode() {
        if (this.isLivePhoneMode) {
            this.stopLivePhoneMode();
            return;
        }

        if (this.isSimulatedMode) {
            this.stopSimulatedPhoneSensors();
            return;
        }

        let permissionRequested = false;

        // 1. iOS 13+ motion permission request (must be called in user gesture event loop)
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            permissionRequested = true;
            DeviceMotionEvent.requestPermission()
                .then(state => {
                    if (state === 'granted') {
                        this.startLivePhoneSensors();
                    } else {
                        if (this.dom.sensorModal) this.dom.sensorModal.classList.remove('hidden');
                    }
                })
                .catch(err => {
                    console.warn('iOS motion permission error:', err);
                    if (this.dom.sensorModal) this.dom.sensorModal.classList.remove('hidden');
                });
        }

        // 2. Geolocation permission prompt (triggers native browser location dialog on Android & iOS)
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    // Native permission granted! Start live tracking
                    this.startLivePhoneSensors();
                },
                (err) => {
                    console.warn('Geolocation permission prompt failed:', err);
                    // If error code is 1 (PERMISSION_DENIED) and insecure context (HTTP), show modal
                    if (!window.isSecureContext && this.dom.sensorModal) {
                        this.dom.sensorModal.classList.remove('hidden');
                    }
                },
                { enableHighAccuracy: true, timeout: 6000, maximumAge: 0 }
            );
        } else {
            if (!permissionRequested) {
                this.startLivePhoneSensors();
            }
        }
    }

    startLivePhoneSensors() {
        this.stopSimulatedPhoneSensors();
        this.isLivePhoneMode = true;
        this.isPlaying = false;
        this.hasReceivedMotion = false;
        this.dom.playIcon.innerText = '▶';
        this.dom.phoneModeBtn.classList.add('active');
        this.dom.phoneModeBtn.innerText = '🔴 Phone Live';
        if (this.dom.sensorLiveReadout) this.dom.sensorLiveReadout.innerText = 'LISTENING...';

        this.motionHandler = (event) => {
            const acc = event.accelerationIncludingGravity || event.acceleration || { x: 0, y: 0, z: 9.81 };
            const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };
            
            const ax = acc.x || 0;
            const ay = acc.y || 0;
            const az = acc.z || 0;
            const gx = (rot.beta || 0) * Math.PI / 180.0;
            const gy = (rot.gamma || 0) * Math.PI / 180.0;
            const gz = (rot.alpha || 0) * Math.PI / 180.0;

            this.hasReceivedMotion = true;
            this.updateWaveforms({ ax, ay, az, gx, gy, gz });

            if (this.dom.sensorLiveReadout) {
                this.dom.sensorLiveReadout.innerText = `Ax:${ax.toFixed(1)} Gz:${gz.toFixed(2)}`;
            }

            // Stationary detector heuristic (ZUPT)
            const normA = Math.hypot(ax, ay, az);
            const isStat = Math.abs(normA - 9.81) < 0.35 && Math.hypot(gx, gy, gz) < 0.05;
            this.updateZupt(isStat ? 0.0 : 4.0);
        };

        this.orientationHandler = (event) => {
            if (event.alpha !== null && event.alpha !== undefined) {
                this.updateCompass(event.alpha);
            }
        };

        window.addEventListener('devicemotion', this.motionHandler);
        window.addEventListener('deviceorientation', this.orientationHandler);

        // Watchdog: detect if browser drops events
        setTimeout(() => {
            if (this.isLivePhoneMode && !this.hasReceivedMotion) {
                if (this.dom.sensorLiveReadout) this.dom.sensorLiveReadout.innerText = 'NO SENSOR DATA';
                if (!window.isSecureContext && this.dom.sensorModal) {
                    this.dom.sensorModal.classList.remove('hidden');
                }
            }
        }, 2500);

        if (navigator.geolocation) {
            this.geoWatchId = navigator.geolocation.watchPosition((pos) => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                let speedKmh = (pos.coords.speed || 0) * 3.6;
                const heading = pos.coords.heading || 0;
                const acc = Math.round(pos.coords.accuracy || 0);

                // Filter indoor GPS multipath jitter:
                // Indoor walls cause phone GPS to hop 10-20 meters randomly, falsely reporting 10-18 km/h.
                // When accuracy is poor (>12m) or speed is low pedestrian jitter, clamp to realistic walking/zero:
                if (acc > 12 && speedKmh < 20) {
                    speedKmh = Math.max(0, speedKmh * 0.2); // dampen indoor multipath
                }
                if (speedKmh < 1.8) {
                    speedKmh = 0.0; // clamp stationary drift
                }

                this.vehicleMarker.setLatLng([lat, lon]);
                this.map.panTo([lat, lon], { animate: true });
                this.updateSpeedometer(speedKmh);
                if (heading) this.updateCompass(heading);
                this.updateModeUI(this.manualOutage ? 'MAP_CONTEXT' : 'GNSS');
                if (this.dom.gnssStatusText) {
                    this.dom.gnssStatusText.innerText = `GPS (±${acc}m)`;
                }
                if (this.dom.locationTagText) {
                    this.dom.locationTagText.innerText = `📱 Live Phone GPS: ${lat.toFixed(4)}, ${lon.toFixed(4)} (±${acc}m)`;
                }
            }, (err) => {
                console.warn('Geo warning:', err);
                if (this.dom.gnssStatusText) {
                    this.dom.gnssStatusText.innerText = 'NO GPS FIX';
                }
            }, { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 });
        }
    }

    stopLivePhoneMode() {
        this.flyToBenchmark();
    }

    flyToBenchmark() {
        this.isLivePhoneMode = false;
        if (this.dom.phoneModeBtn) {
            this.dom.phoneModeBtn.classList.remove('active');
            this.dom.phoneModeBtn.innerText = '📱 Live Sensors';
        }
        if (this.dom.sensorLiveReadout) this.dom.sensorLiveReadout.innerText = 'STANDBY';

        this.stopSimulatedPhoneSensors();

        if (this.motionHandler) window.removeEventListener('devicemotion', this.motionHandler);
        if (this.orientationHandler) window.removeEventListener('deviceorientation', this.orientationHandler);
        if (this.geoWatchId && navigator.geolocation) {
            navigator.geolocation.clearWatch(this.geoWatchId);
            this.geoWatchId = null;
        }

        if (this.userAccCircle) {
            this.map.removeLayer(this.userAccCircle);
            this.userAccCircle = null;
        }

        if (this.dom.btnLocateMe) {
            this.dom.btnLocateMe.classList.remove('active');
            this.dom.btnLocateMe.innerText = '📍 My Location';
        }

        // Reset manual outage if active
        if (this.manualOutage) {
            this.manualOutage = false;
            if (this.dom.toggleOutageBtn) {
                this.dom.toggleOutageBtn.innerText = 'Inject Outage';
                this.dom.toggleOutageBtn.style.background = 'rgba(239, 68, 68, 0.15)';
                this.dom.toggleOutageBtn.style.color = '#ef4444';
            }
        }
        if (this.dom.blackoutBanner) {
            this.dom.blackoutBanner.classList.add('hidden');
        }

        if (this.dom.locationTagText) {
            this.dom.locationTagText.innerText = '🚗 Demo Replay: Coventry Benchmark (UK)';
        }

        if (this.dom.storyDesc) {
            this.dom.storyDesc.innerHTML = '<strong>Coventry Benchmark Replay</strong> active. Driving along test route under normal GPS conditions.';
        }

        // Button click visual feedback
        const btnBenchmark = document.getElementById('btnBenchmarkRoute');
        if (btnBenchmark) {
            btnBenchmark.innerText = '🚗 Benchmark Active!';
            btnBenchmark.classList.add('active');
            setTimeout(() => {
                const btn = document.getElementById('btnBenchmarkRoute');
                if (btn) {
                    btn.innerText = '🚗 Benchmark Route';
                    btn.classList.remove('active');
                }
            }, 1200);
        }

        // Reset replay timeline to start if at or near end so user immediately sees vehicle moving
        if (this.data && this.data.frames && this.data.frames.length > 0) {
            if (!this.currentIndex || this.currentIndex >= this.data.frames.length - 20) {
                this.currentIndex = 0;
            }
            if (this.dom.timelineSlider) {
                this.dom.timelineSlider.value = this.currentIndex;
            }

            const frame = this.data.frames[this.currentIndex];
            const targetPos = frame.mode === 'GNSS' ? frame.gt.slice(0, 2) : frame.fused;

            if (this.vehicleMarker) {
                this.vehicleMarker.setLatLng(targetPos);
            }

            // Using setView guarantees instant, error-free camera snap back to Coventry UK
            if (this.map) {
                this.map.setView(targetPos, 16);
                setTimeout(() => {
                    if (this.map) this.map.invalidateSize();
                }, 100);
            }

            this.renderFrame(this.currentIndex);
            this.isPlaying = true;
            if (this.dom.playIcon) this.dom.playIcon.innerText = '❚❚';
        } else if (this.map) {
            // Fallback default coordinates for Coventry, UK
            this.map.setView([52.5618, -1.4552], 16);
        }
    }

    locateUser() {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser.');
            return;
        }

        if (this.dom.btnLocateMe) {
            this.dom.btnLocateMe.innerText = '📍 Locating...';
        }

        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                const acc = Math.round(pos.coords.accuracy || 0);

                if (this.dom.btnLocateMe) {
                    this.dom.btnLocateMe.innerText = '📍 My Location';
                    this.dom.btnLocateMe.classList.add('active');
                }

                if (this.dom.locationTagText) {
                    this.dom.locationTagText.innerText = `📍 Your Location: ${lat.toFixed(4)}, ${lon.toFixed(4)} (±${acc}m)`;
                }

                if (this.dom.storyDesc) {
                    this.dom.storyDesc.innerHTML = `<strong>Located your device!</strong> Found GPS fix (±${acc}m). Map flew to your real location. Tap <strong>"Live Sensors"</strong> to stream motion!`;
                }

                // Pause benchmark replay
                this.isPlaying = false;
                if (this.dom.playIcon) this.dom.playIcon.innerText = '▶';

                // Move vehicle marker and fly map
                this.vehicleMarker.setLatLng([lat, lon]);
                this.map.flyTo([lat, lon], 17, { duration: 1.5 });

                if (this.userAccCircle) {
                    this.map.removeLayer(this.userAccCircle);
                }
                this.userAccCircle = L.circle([lat, lon], {
                    radius: Math.max(10, acc),
                    color: '#00f0ff',
                    fillColor: '#00f0ff',
                    fillOpacity: 0.18,
                    weight: 2
                }).addTo(this.map);
            },
            (err) => {
                console.warn('Locate error:', err);
                if (this.dom.btnLocateMe) {
                    this.dom.btnLocateMe.innerText = '📍 My Location';
                }
                if (!window.isSecureContext) {
                    if (this.dom.sensorModal) this.dom.sensorModal.classList.remove('hidden');
                } else {
                    alert('Could not retrieve location: ' + (err.message || 'Permission denied'));
                }
            },
            { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }
        );
    }

    startSimulatedPhoneSensors() {
        this.stopLivePhoneMode();
        this.isSimulatedMode = true;
        this.isPlaying = true;
        this.dom.playIcon.innerText = '❚❚';
        this.dom.phoneModeBtn.classList.add('active');
        this.dom.phoneModeBtn.innerText = '🎮 Sim Active';
        if (this.dom.sensorLiveReadout) this.dom.sensorLiveReadout.innerText = 'SIM ACTIVE';
    }

    stopSimulatedPhoneSensors() {
        this.isSimulatedMode = false;
        this.dom.phoneModeBtn.classList.remove('active');
        this.dom.phoneModeBtn.innerText = '📱 Live Sensors';
        if (this.dom.sensorLiveReadout) this.dom.sensorLiveReadout.innerText = 'STANDBY';
    }

    startLoop() {
        let lastTime = performance.now();
        let frameAccum = 0;

        const loop = (currentTime) => {
            const dt = (currentTime - lastTime) / 1000.0;
            lastTime = currentTime;

            if (this.isPlaying && this.data && this.data.frames) {
                // At 10Hz, 1 frame is 0.1s
                frameAccum += dt * this.playbackSpeed * 10.0;
                while (frameAccum >= 1.0) {
                    if (this.currentIndex < this.data.frames.length - 1) {
                        this.currentIndex++;
                        frameAccum -= 1.0;
                    } else {
                        this.isPlaying = false;
                        this.dom.playIcon.innerText = '▶';
                        break;
                    }
                }
                this.dom.timelineSlider.value = this.currentIndex;
                this.renderFrame(this.currentIndex);
            }

            requestAnimationFrame(loop);
        };

        requestAnimationFrame(loop);
    }

    renderFrame(idx) {
        if (!this.data || !this.data.frames || idx >= this.data.frames.length) return;
        const frame = this.data.frames[idx];

        // 1. Time display
        this.dom.timeElapsed.innerText = this.formatTime(frame.t);

        // 2. Mode resolution (including manual outage injection)
        let effectiveMode = frame.mode;
        if (this.manualOutage) {
            effectiveMode = 'MAP_CONTEXT';
        }

        this.updateModeUI(effectiveMode);

        // 3. Update Vehicle Marker & Trajectories
        const pos = effectiveMode === 'GNSS' ? frame.gt.slice(0, 2) : frame.fused;
        this.vehicleMarker.setLatLng(pos);

        // Rotate arrow to heading
        const headingDeg = frame.gt[2];
        const arrow = document.getElementById('vehicleArrow');
        if (arrow) {
            arrow.style.transform = `rotate(${headingDeg}deg)`;
        }

        // Draw path history (last 200 points)
        const startWindow = Math.max(0, idx - 200);
        const history = this.data.frames.slice(startWindow, idx + 1);

        this.gtPolyline.setLatLngs(history.map(f => f.gt.slice(0, 2)));
        this.fusedPolyline.setLatLngs(history.map(f => f.fused));

        if (effectiveMode !== 'GNSS') {
            this.rawDrPolyline.setLatLngs(history.map(f => f.raw_dr));
        } else {
            this.rawDrPolyline.setLatLngs([]);
        }

        // Pan map smoothly if moving near edge
        if (idx % 10 === 0) {
            this.map.panTo(pos, { animate: true, duration: 0.5 });
        }

        // 4. Update Gauges & GNSS Flow Hero Values
        this.currentSpeed = frame.speed_kmh;
        if (this.flowState === 1 && this.dom.flowCardHero) {
            this.dom.flowCardHero.innerHTML = `${Math.round(frame.speed_kmh)} <span class="hero-unit">km/h</span>`;
        } else if (this.flowState === 3) {
            this.blackoutElapsedSecs += 0.1 * this.playbackSpeed;
            const timerEl = document.getElementById('flowBlackoutTimer');
            if (timerEl) {
                timerEl.innerText = this.formatBlackoutTime(this.blackoutElapsedSecs);
            }
        }

        this.updateSpeedometer(frame.speed_kmh);
        this.updateZupt(frame.speed_mps);
        this.updateCompass(headingDeg);

        // 5. Update Metrics HUD
        this.dom.errorVal.innerText = `${frame.err_m} m`;
        const drift = Math.min(99.9, (frame.err_m / Math.max(10, frame.t * 10)) * 100).toFixed(1);
        this.dom.driftVal.innerText = `${drift} %`;

        // 6. Update Oscilloscope Waveforms
        this.updateWaveforms(frame.imu);
    }

    updateModeUI(mode) {
        this.dom.modeValue.innerText = mode;

        if (mode === 'GNSS') {
            this.dom.modeDot.style.background = 'var(--neon-emerald)';
            this.dom.modeDot.style.boxShadow = '0 0 10px var(--neon-emerald)';
            this.dom.gnssStatusText.innerText = 'LOCKED (12 SV)';
            this.dom.gnssBars.forEach(b => b.classList.add('active'));
            this.dom.mapStatusPill.innerText = 'PASS-THROUGH';
            if (this.dom.blackoutBanner) {
                this.dom.blackoutBanner.classList.remove('user-dismissed');
                this.dom.blackoutBanner.classList.add('hidden');
            }

            if (this.dom.storyBadge && this.dom.storyModeLabel && this.dom.storyDesc) {
                this.dom.storyBadge.innerText = '🟢 GPS LOCKED';
                this.dom.storyBadge.classList.remove('outage');
                this.dom.storyModeLabel.innerText = 'Normal Road Driving';
                this.dom.storyDesc.innerHTML = 'Driving under clear sky with GPS satellites active. Tap <strong>"Inject Outage"</strong> below to simulate driving into a tunnel!';
            }
        } else if (mode === 'MAP_CONTEXT') {
            this.dom.modeDot.style.background = 'var(--neon-cyan)';
            this.dom.modeDot.style.boxShadow = '0 0 10px var(--neon-cyan)';
            this.dom.gnssStatusText.innerText = 'BLACKOUT (0 SV)';
            this.dom.gnssBars.forEach(b => b.classList.remove('active'));
            this.dom.mapStatusPill.innerText = 'HMM SNAPPED';
            if (this.dom.blackoutBanner && !this.dom.blackoutBanner.classList.contains('user-dismissed')) {
                this.dom.blackoutBanner.classList.remove('hidden');
            }

            if (this.dom.storyBadge && this.dom.storyModeLabel && this.dom.storyDesc) {
                this.dom.storyBadge.innerText = '⚠️ TUNNEL MODE';
                this.dom.storyBadge.classList.add('outage');
                this.dom.storyModeLabel.innerText = 'NAVIK AI Dead-Reckoning';
                this.dom.storyDesc.innerHTML = '<strong>GPS Signal Lost!</strong> Regular navigation would freeze here. NAVIK AI is predicting vehicle velocity and snapping your car to the road!';
            }
        } else if (mode === 'DEAD_RECKONING') {
            this.dom.modeDot.style.background = 'var(--neon-amber)';
            this.dom.modeDot.style.boxShadow = '0 0 10px var(--neon-amber)';
            this.dom.gnssStatusText.innerText = 'BLACKOUT (0 SV)';
            this.dom.gnssBars.forEach(b => b.classList.remove('active'));
            this.dom.mapStatusPill.innerText = 'SEARCHING';
            if (this.dom.blackoutBanner && !this.dom.blackoutBanner.classList.contains('user-dismissed')) {
                this.dom.blackoutBanner.classList.remove('hidden');
            }

            if (this.dom.storyBadge && this.dom.storyModeLabel && this.dom.storyDesc) {
                this.dom.storyBadge.innerText = '⚠️ INERTIAL DR';
                this.dom.storyBadge.classList.add('outage');
                this.dom.storyModeLabel.innerText = 'Inertial Dead-Reckoning';
                this.dom.storyDesc.innerHTML = 'Off-road or searching for road match. Tracking vehicle motion purely via 6-axis IMU SpeedNet integration.';
            }
        } else {
            // DR_FALLBACK
            this.dom.modeDot.style.background = 'var(--neon-crimson)';
            this.dom.modeDot.style.boxShadow = '0 0 10px var(--neon-crimson)';
            this.dom.gnssStatusText.innerText = 'BLACKOUT (0 SV)';
            this.dom.gnssBars.forEach(b => b.classList.remove('active'));
            this.dom.mapStatusPill.innerText = 'DR FALLBACK';
            if (this.dom.storyBadge && this.dom.storyModeLabel && this.dom.storyDesc) {
                this.dom.storyBadge.innerText = '🚨 DR FALLBACK';
                this.dom.storyBadge.classList.add('outage');
                this.dom.storyModeLabel.innerText = 'Extended Outage';
                this.dom.storyDesc.innerHTML = 'Prolonged GPS outage (>100m). Kalman filter covariance growing; maintaining vehicle state estimates.';
            }
        }
    }

    setFlowState(stateNum, autoTriggered = false) {
        this.flowState = stateNum;

        // Update Stepper buttons active state
        if (this.dom.flowStepBtns) {
            this.dom.flowStepBtns.forEach(btn => {
                btn.classList.toggle('active', parseInt(btn.dataset.step, 10) === stateNum);
            });
        }

        // Cancel auto-transition timers if user clicked manually
        if (!autoTriggered && this.flowTransitionTimer) {
            clearTimeout(this.flowTransitionTimer);
            this.flowTransitionTimer = null;
        }

        const pill = this.dom.flowStatusPill;
        const banner = this.dom.flowSlideBanner;
        const confBadge = this.dom.flowConfidenceBadge;
        const bottomCard = this.dom.flowBottomCard;
        const cardLabel = this.dom.flowCardLabel;
        const cardHero = this.dom.flowCardHero;

        this.updateMarkerVisual(stateNum);

        if (stateNum === 1) {
            // 1. GNSS active
            if (pill) {
                pill.className = 'flow-status-pill state-1';
                this.dom.flowPillIcon.innerText = '🛰️';
                this.dom.flowPillText.innerText = 'GNSS active';
            }
            if (banner) banner.classList.add('hidden');
            if (confBadge) confBadge.classList.add('hidden');

            if (bottomCard) {
                bottomCard.className = 'cockpit-hero-row flow-bottom-card state-1';
                if (cardLabel) cardLabel.innerText = 'SPEED';
                if (cardHero) cardHero.innerHTML = `${Math.round(this.currentSpeed || 42)} <span class="hero-unit">km/h</span>`;
            }

            if (this.fusedPolyline) {
                this.fusedPolyline.setStyle({ color: '#00f0ff', dashArray: null, weight: 4 });
            }
            if (this.candidateRoadPoly) this.candidateRoadPoly.setLatLngs([]);
            if (this.connectorPoly) this.connectorPoly.setLatLngs([]);
            if (this.connectorDot && this.map && this.map.hasLayer(this.connectorDot)) {
                this.map.removeLayer(this.connectorDot);
            }

            this.updateModeUI('GNSS');
        } else if (stateNum === 2) {
            // 2. Signal lost
            if (pill) {
                pill.className = 'flow-status-pill state-2';
                this.dom.flowPillIcon.innerText = '⚠️';
                this.dom.flowPillText.innerText = 'Signal lost';
            }
            if (banner) {
                banner.classList.remove('hidden');
                setTimeout(() => {
                    if (banner) banner.classList.add('hidden');
                }, 4000);
            }
            if (confBadge) confBadge.classList.add('hidden');

            if (bottomCard) {
                bottomCard.className = 'cockpit-hero-row flow-bottom-card state-2';
                if (cardLabel) cardLabel.innerText = 'SWITCHING TO';
                if (cardHero) cardHero.innerHTML = `<span style="color:#f59e0b">Dead reckoning</span>`;
            }

            if (this.fusedPolyline) {
                this.fusedPolyline.setStyle({ color: '#f59e0b', dashArray: '8, 8', weight: 4 });
            }
            if (this.candidateRoadPoly) this.candidateRoadPoly.setLatLngs([]);
            if (this.connectorPoly) this.connectorPoly.setLatLngs([]);
            if (this.connectorDot && this.map && this.map.hasLayer(this.connectorDot)) {
                this.map.removeLayer(this.connectorDot);
            }

            this.updateModeUI('DEAD_RECKONING');

            if (autoTriggered) {
                this.flowTransitionTimer = setTimeout(() => {
                    this.setFlowState(3, true);
                }, 1800);
            }
        } else if (stateNum === 3) {
            // 3. DR + map
            if (pill) {
                pill.className = 'flow-status-pill state-3';
                this.dom.flowPillIcon.innerText = '📍';
                this.dom.flowPillText.innerText = 'DR + map';
            }
            if (banner) banner.classList.add('hidden');
            if (confBadge) {
                confBadge.classList.remove('hidden');
                if (this.dom.flowConfNum) this.dom.flowConfNum.innerText = '72%';
            }

            if (bottomCard) {
                bottomCard.className = 'cockpit-hero-row flow-bottom-card state-3';
                if (cardLabel) cardLabel.innerText = 'BLACKOUT TIME';
                if (cardHero) cardHero.innerHTML = `<span style="color:#f59e0b" id="flowBlackoutTimer">${this.formatBlackoutTime(this.blackoutElapsedSecs)}</span>`;
            }

            if (this.fusedPolyline) {
                this.fusedPolyline.setStyle({ color: '#f59e0b', dashArray: '8, 8', weight: 4 });
            }

            // Draw candidate road segment and snap connector dot
            if (this.vehicleMarker) {
                const curPos = this.vehicleMarker.getLatLng();
                const offsetLat = 0.00025;
                const offsetLon = 0.00035;
                const roadStart = [curPos.lat - offsetLat, curPos.lng + offsetLon - 0.0004];
                const roadSnap = [curPos.lat + 0.00008, curPos.lng + offsetLon];
                const roadEnd = [curPos.lat + offsetLat + 0.0002, curPos.lng + offsetLon + 0.0004];

                if (this.candidateRoadPoly) {
                    this.candidateRoadPoly.setLatLngs([roadStart, roadSnap, roadEnd]);
                }
                if (this.connectorPoly) {
                    this.connectorPoly.setLatLngs([[curPos.lat, curPos.lng], roadSnap]);
                }
                if (this.connectorDot && this.map) {
                    this.connectorDot.setLatLng(roadSnap).addTo(this.map);
                }
            }

            this.updateModeUI('MAP_CONTEXT');
        } else if (stateNum === 4) {
            // 4. GNSS restored
            if (pill) {
                pill.className = 'flow-status-pill state-4';
                this.dom.flowPillIcon.innerText = '✓';
                this.dom.flowPillText.innerText = 'Restored';
            }
            if (banner) banner.classList.add('hidden');
            if (confBadge) confBadge.classList.add('hidden');

            if (bottomCard) {
                bottomCard.className = 'cockpit-hero-row flow-bottom-card state-4';
                if (cardLabel) cardLabel.innerText = 'POSITION';
                if (cardHero) cardHero.innerHTML = `<span style="color:#10b981">Corrected</span>`;
            }

            if (this.fusedPolyline) {
                this.fusedPolyline.setStyle({ color: '#10b981', dashArray: null, weight: 4 });
            }
            if (this.candidateRoadPoly) this.candidateRoadPoly.setLatLngs([]);
            if (this.connectorPoly) this.connectorPoly.setLatLngs([]);
            if (this.connectorDot && this.map && this.map.hasLayer(this.connectorDot)) {
                this.map.removeLayer(this.connectorDot);
            }

            this.updateModeUI('GNSS');

            if (autoTriggered) {
                this.flowTransitionTimer = setTimeout(() => {
                    this.setFlowState(1, true);
                }, 2800);
            }
        }
    }

    updateMarkerVisual(stateNum) {
        if (!this.vehicleMarker) return;
        let html = '';
        if (stateNum === 1) {
            html = `
                <div class="pos-marker-container state-1">
                    <div style="position:absolute; width:26px; height:26px; border-radius:50%; background:rgba(0,240,255,0.25);"></div>
                    <div class="pos-marker-dot"></div>
                    <div id="vehicleArrow" style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:16px solid #00f0ff; position:absolute; top:-12px; filter:drop-shadow(0 0 4px #00f0ff); transform-origin:50% 20px;"></div>
                </div>
            `;
        } else if (stateNum === 2) {
            html = `
                <div class="pos-marker-container state-2">
                    <div class="pulse-estimating-ring"></div>
                    <div class="pos-marker-dot" style="background:#f59e0b; border-color:#fff; box-shadow:0 0 12px #f59e0b;"></div>
                    <div id="vehicleArrow" style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:16px solid #f59e0b; position:absolute; top:-12px; filter:drop-shadow(0 0 4px #f59e0b); transform-origin:50% 20px;"></div>
                </div>
            `;
        } else if (stateNum === 3) {
            html = `
                <div class="pos-marker-container state-3">
                    <div class="pos-marker-dot" style="background:#f59e0b; border-color:#fff; box-shadow:0 0 14px #f59e0b;"></div>
                    <div id="vehicleArrow" style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:16px solid #f59e0b; position:absolute; top:-12px; filter:drop-shadow(0 0 4px #f59e0b); transform-origin:50% 20px;"></div>
                </div>
            `;
        } else if (stateNum === 4) {
            html = `
                <div class="pos-marker-container state-4">
                    <div class="pulse-resync-ring"></div>
                    <div class="pos-marker-dot" style="background:#10b981; border-color:#fff; box-shadow:0 0 14px #10b981;"></div>
                    <div id="vehicleArrow" style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:16px solid #10b981; position:absolute; top:-12px; filter:drop-shadow(0 0 4px #10b981); transform-origin:50% 20px;"></div>
                </div>
            `;
        }

        const newIcon = L.divIcon({
            className: 'vehicle-marker-icon',
            html: html,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        });
        this.vehicleMarker.setIcon(newIcon);
    }

    formatBlackoutTime(totalSecs) {
        const mins = Math.floor(totalSecs / 60);
        const secs = Math.floor(totalSecs % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    getCardinal(deg) {
        const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
        const idx = Math.round(((deg % 360) / 22.5)) % 16;
        return directions[idx];
    }

    updateSpeedometer(kmh) {
        this.dom.speedValue.innerText = kmh.toFixed(1);
        const hudSpeed = document.getElementById('hudSpeedVal');
        if (hudSpeed) hudSpeed.innerText = kmh.toFixed(1);

        // Gauge circle circumference = 2 * PI * 50 ≈ 314
        const maxKmh = 100.0;
        const fraction = Math.min(1.0, kmh / maxKmh);
        const offset = 314 - (fraction * 314 * 0.75); // 270 degree arc
        this.dom.speedGaugeCircle.style.strokeDashoffset = offset;
    }

    updateZupt(speedMps) {
        // Stationary if speed < 0.2 m/s
        const isStationary = speedMps < 0.2;
        const prob = isStationary ? 0.94 : Math.max(0.01, 0.15 - (speedMps * 0.02));
        
        this.dom.zuptMeterFill.style.height = `${(prob * 100).toFixed(0)}%`;
        this.dom.zuptProb.innerText = `${prob.toFixed(2)} P`;
        this.dom.zuptState.innerText = isStationary ? 'STATIONARY (ZUPT)' : 'MOVING';
        this.dom.zuptState.style.color = isStationary ? 'var(--neon-crimson)' : 'var(--neon-emerald)';

        const hudZupt = document.getElementById('hudZuptVal');
        if (hudZupt) {
            hudZupt.innerText = isStationary ? 'STOPPED' : 'MOVING';
            hudZupt.style.color = isStationary ? 'var(--neon-crimson)' : 'var(--neon-emerald)';
        }
    }

    updateCompass(deg) {
        const degStr = `${Math.round(deg).toString().padStart(3, '0')}°`;
        this.dom.headingReadout.innerText = degStr;
        this.dom.compassDial.style.transform = `rotate(${-deg}deg)`;

        const hudHeading = document.getElementById('hudHeadingVal');
        if (hudHeading) hudHeading.innerText = degStr;
        const hudDir = document.getElementById('hudHeadingDir');
        if (hudDir) hudDir.innerText = this.getCardinal(deg);
    }

    updateWaveforms(imu) {
        this.accelHistory.x.push(imu.ax);
        this.accelHistory.y.push(imu.ay);
        this.accelHistory.z.push(imu.az);

        this.gyroHistory.x.push(imu.gx);
        this.gyroHistory.y.push(imu.gy);
        this.gyroHistory.z.push(imu.gz);

        if (this.accelHistory.x.length > this.maxWaveformPoints) {
            this.accelHistory.x.shift();
            this.accelHistory.y.shift();
            this.accelHistory.z.shift();

            this.gyroHistory.x.shift();
            this.gyroHistory.y.shift();
            this.gyroHistory.z.shift();
        }

        this.drawWaveform(this.dom.accelCanvas, this.accelHistory, -12, 12);
        this.drawWaveform(this.dom.gyroCanvas, this.gyroHistory, -0.5, 0.5);
    }

    drawWaveform(canvas, history, minVal, maxVal) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Center zero line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        const zeroY = h - ((0 - minVal) / (maxVal - minVal)) * h;
        ctx.moveTo(0, zeroY);
        ctx.lineTo(w, zeroY);
        ctx.stroke();

        const channels = [
            { data: history.x, color: '#f43f5e' },
            { data: history.y, color: '#10b981' },
            { data: history.z, color: '#00f0ff' }
        ];

        channels.forEach(ch => {
            if (ch.data.length < 2) return;
            ctx.strokeStyle = ch.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();

            const stepX = w / (this.maxWaveformPoints - 1);
            for (let i = 0; i < ch.data.length; i++) {
                const x = i * stepX;
                const normalized = (ch.data[i] - minVal) / (maxVal - minVal);
                const y = Math.max(2, Math.min(h - 2, h - normalized * h));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
    }
}

// Instantiate on load
window.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NavikDashboard();
});
