package com.signlanguage.sign_translator_backend;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

@RestController
public class SignPredictionController {

    @Autowired
    private RestTemplate restTemplate;

    private static final String FLASK_URL = "http://127.0.0.1:5000/predict";

    @GetMapping("/api/health")
    public ResponseEntity<Map<String, Object>> health() {
        try {
            Map response = restTemplate.getForObject(FLASK_URL.replace("/predict", "/health"), Map.class);
            return ResponseEntity.ok(response);
        } catch (RestClientException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("status", "unavailable", "error", "Python recognition service is not running"));
        }
    }

    @PostMapping("/api/predict-sign")
    public ResponseEntity<Map<String, Object>> predictSign(@RequestBody Map<String, List<List<Double>>> request) {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, List<List<Double>>>> entity = new HttpEntity<>(request, headers);

        try {
            Map response = restTemplate.postForObject(FLASK_URL, entity, Map.class);
            return ResponseEntity.ok(response);
        } catch (RestClientException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Python recognition service is not running"));
        }
    }
}
