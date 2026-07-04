Which Techniques Are Used Most?
Technique	Used in Industry?	Why
Rule-based checks	✅ Very common	Fast, simple, and effective for known patterns
Duplicate matching	✅ Standard	Prevents accidental and intentional double payments
Three-way matching	✅ Standard	Core procurement control
Statistical analysis	✅ Common	Detects unusual values and trends
Vendor risk scoring	✅ Common	Prioritizes high-risk vendors
Machine learning	✅ Increasingly common	Finds subtle and previously unknown anomalies
Network analysis	✅ Used by larger organizations	Effective for detecting collusion and shell vendors
NLP	⚠️ Growing use	Useful for comparing invoice descriptions
Benford's Law	⚠️ Used by auditors	Helps identify datasets worth investigating


                     Fraud Detection System

                            |
    -------------------------------------------------------
    |         |          |         |         |            |
 Invoice    Vendor      PO       GRN      Payments    Employees
    |         |          |         |         |            |
    -------------------------------------------------------
                            |
                  Feature Engineering
                            |
                 Rule Engine + ML Models
                            |
                     Fraud Risk Score

<!-- Invoice data=invoice id,number,vendor id,invoice date,amount, -->

db schema-->file:///C:/Users/Lavanya/Downloads/invoice_anomaly_detection_schema.html

Welcome to your new dbt project!

### Using the starter project

Try running the following commands:
- dbt run
- dbt test


### Resources:
- Learn more about dbt [in the docs](https://docs.getdbt.com/docs/introduction)
- Check out [Discourse](https://discourse.getdbt.com/) for commonly asked questions and answers
- Join the [chat](https://community.getdbt.com/) on Slack for live discussions and support
- Find [dbt events](https://events.getdbt.com) near you
- Check out [the blog](https://blog.getdbt.com/) for the latest news on dbt's development and best practices

