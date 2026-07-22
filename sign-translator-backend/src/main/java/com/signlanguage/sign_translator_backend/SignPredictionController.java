package com.signlanguage.sign_translator_backend;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class SignPredictionController {

    @Autowired
    private RestTemplate restTemplate;

    private static final String FLASK_URL = "http://127.0.0.1:5000/predict";

    @PostMapping("/api/predict-sign")
    public Map predictSign(@RequestBody Map<String, List<List<Double>>> request) {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, List<List<Double>>>> entity = new HttpEntity<>(request, headers);

        Map response = restTemplate.postForObject(FLASK_URL, entity, Map.class);

        return response;
    }
}