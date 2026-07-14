# data_generation/config.py

VENDOR_COUNT = 500
INVOICE_COUNT = 50000
START_YEAR = 2025

CURRENCY = "INR"

PAYMENT_TERMS = [15, 30, 45, 60]

VENDOR_CATEGORIES = {
    "Office Supplies": {
        "mu": 8.2,
        "sigma": 0.45
    },
    "IT Equipment": {
        "mu": 11.2,
        "sigma": 0.60
    },
    "Software": {
        "mu": 10.5,
        "sigma": 0.50
    },
    "Manufacturing": {
        "mu": 11.6,
        "sigma": 0.55
    },
    "Logistics": {
        "mu": 10.0,
        "sigma": 0.50
    },
    "Maintenance": {
        "mu": 9.7,
        "sigma": 0.45
    },
    "Consulting": {
        "mu": 10.8,
        "sigma": 0.55
    },
    "Furniture": {
        "mu": 10.3,
        "sigma": 0.50
    }
}