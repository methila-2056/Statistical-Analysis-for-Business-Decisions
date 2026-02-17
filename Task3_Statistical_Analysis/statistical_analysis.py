import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysis:
    """Class to perform statistical analysis for business decisions"""
    
    def __init__(self, data_path):
        """Initialize with dataset"""
        self.df = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully!")
        print(f"  Rows: {len(self.df)}")
        print(f"  Columns: {len(self.df.columns)}")
        print(f"\nColumns: {list(self.df.columns)}")
    
    def data_overview(self):
        """Display data overview"""
        print("\n" + "="*80)
        print("DATA OVERVIEW")
        print("="*80)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\n\nData Types:")
        print(self.df.dtypes)
        
        print("\n\nBasic Statistics:")
        print(self.df.describe())
        
        print("\n\nMissing Values:")
        print(self.df.isnull().sum())
    
    def hypothesis_testing_ttest(self):
        """Perform t-test hypothesis testing"""
        print("\n" + "="*80)
        print("1. HYPOTHESIS TESTING - T-TEST")
        print("="*80)
        
        print("\nBusiness Question: Do customers who churn have different charges than those who don't?")
        print("\nH0 (Null Hypothesis): No difference in average charges between churned and non-churned customers")
        print("H1 (Alternative Hypothesis): There IS a difference in charges")
        
        self.df['Total_Charges'] = (self.df['Total day charge'] + 
                                    self.df['Total eve charge'] + 
                                    self.df['Total night charge'] + 
                                    self.df['Total intl charge'])
    
        churned = self.df[self.df['Churn'] == True]['Total_Charges']
        not_churned = self.df[self.df['Churn'] == False]['Total_Charges']

        t_statistic, p_value = ttest_ind(churned, not_churned)
        
        print(f"\n📊 T-Test Results:")
        print(f"{'─'*60}")
        print(f"Churned customers - Mean charge: ${churned.mean():.2f}")
        print(f"Not churned customers - Mean charge: ${not_churned.mean():.2f}")
        print(f"Difference: ${abs(churned.mean() - not_churned.mean()):.2f}")
        print(f"\nT-statistic: {t_statistic:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significance level (α): 0.05")
        
        if p_value < 0.05:
            print(f"\n✓ RESULT: REJECT null hypothesis (p < 0.05)")
            print(f"  There IS a statistically significant difference in charges!")
        else:
            print(f"\n✗ RESULT: FAIL TO REJECT null hypothesis (p >= 0.05)")
            print(f"  No significant difference in charges")

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(churned, bins=30, alpha=0.7, label='Churned', color='red')
        plt.hist(not_churned, bins=30, alpha=0.7, label='Not Churned', color='green')
        plt.xlabel('Total Charges ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Charges: Churned vs Not Churned')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        box_data = [not_churned, churned]
        plt.boxplot(box_data, labels=['Not Churned', 'Churned'])
        plt.ylabel('Total Charges ($)')
        plt.title('Boxplot Comparison')
        plt.tight_layout()
        plt.savefig('visualizations/ttest_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: visualizations/ttest_visualization.png")
        plt.close()
    
    def hypothesis_testing_chi_square(self):
        """Perform chi-square test"""
        print("\n" + "="*80)
        print("2. HYPOTHESIS TESTING - CHI-SQUARE TEST")
        print("="*80)
        
        print("\nBusiness Question: Is there a relationship between International Plan and Churn?")
        print("\nH0 (Null Hypothesis): International Plan and Churn are independent")
        print("H1 (Alternative Hypothesis): There IS a relationship between them")

        contingency_table = pd.crosstab(self.df['International plan'], self.df['Churn'])
        
        print("\n📊 Contingency Table:")
        print(contingency_table)

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n📊 Chi-Square Test Results:")
        print(f"{'─'*60}")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Significance level (α): 0.05")
        
        if p_value < 0.05:
            print(f"\n✓ RESULT: REJECT null hypothesis (p < 0.05)")
            print(f"  There IS a significant relationship between International Plan and Churn!")
        else:
            print(f"\n✗ RESULT: FAIL TO REJECT null hypothesis (p >= 0.05)")
            print(f"  No significant relationship")
        
        print("\n📊 Churn Rate by International Plan:")
        churn_rate = self.df.groupby('International plan')['Churn'].apply(lambda x: (x == True).sum() / len(x) * 100)
        for plan, rate in churn_rate.items():
            print(f"  {plan}: {rate:.2f}%")
 
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        contingency_table.plot(kind='bar', stacked=False)
        plt.title('Churn by International Plan')
        plt.xlabel('International Plan')
        plt.ylabel('Count')
        plt.legend(title='Churn')
        
        plt.subplot(1, 2, 2)
        churn_rate.plot(kind='bar', color=['green', 'red'])
        plt.title('Churn Rate by International Plan')
        plt.xlabel('International Plan')
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('visualizations/chisquare_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: visualizations/chisquare_visualization.png")
        plt.close()
    
    def ab_testing(self):
        """Perform A/B testing simulation"""
        print("\n" + "="*80)
        print("3. A/B TESTING FOR MARKETING CAMPAIGNS")
        print("="*80)
        
        print("\nBusiness Scenario: Testing two marketing campaigns")
        print("  Campaign A (Control): Standard marketing")
        print("  Campaign B (Treatment): New marketing strategy with incentives")

        group_a = self.df[self.df['Voice mail plan'] == 'No']['Churn']
        group_b = self.df[self.df['Voice mail plan'] == 'Yes']['Churn']
  
        churn_a = (group_a == True).sum() / len(group_a) * 100
        churn_b = (group_b == True).sum() / len(group_b) * 100
        
        print(f"\n📊 Campaign Results:")
        print(f"{'─'*60}")
        print(f"Campaign A (Control):")
        print(f"  Sample size: {len(group_a)}")
        print(f"  Churn rate: {churn_a:.2f}%")
        print(f"\nCampaign B (Treatment):")
        print(f"  Sample size: {len(group_b)}")
        print(f"  Churn rate: {churn_b:.2f}%")
        print(f"\nDifference: {abs(churn_a - churn_b):.2f}%")
        
        t_stat, p_val = ttest_ind(group_a == True, group_b == True)
        
        print(f"\n📊 Statistical Significance Test:")
        print(f"{'─'*60}")
        print(f"P-value: {p_val:.6f}")
        
        if p_val < 0.05:
            print(f"\n✓ RESULT: Campaign B is SIGNIFICANTLY better than Campaign A!")
            print(f"  Recommendation: Implement Campaign B")
        else:
            print(f"\n✗ RESULT: No significant difference between campaigns")
            print(f"  Recommendation: Continue testing or use either campaign")

        plt.figure(figsize=(10, 6))
        
        campaigns = ['Campaign A\n(Control)', 'Campaign B\n(Treatment)']
        churn_rates = [churn_a, churn_b]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = plt.bar(campaigns, churn_rates, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Churn Rate (%)', fontsize=12)
        plt.title('A/B Test Results: Churn Rate Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, max(churn_rates) * 1.2)

        for bar, rate in zip(bars, churn_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/ab_testing_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: visualizations/ab_testing_visualization.png")
        plt.close()
    
    def confidence_intervals(self):
        """Calculate confidence intervals"""
        print("\n" + "="*80)
        print("4. CONFIDENCE INTERVALS & MARGIN OF ERROR")
        print("="*80)
        
        print("\nBusiness Question: What is the average customer charge with 95% confidence?")
 
        charges = (self.df['Total day charge'] + 
                  self.df['Total eve charge'] + 
                  self.df['Total night charge'] + 
                  self.df['Total intl charge'])

        mean_charge = charges.mean()
        std_charge = charges.std()
        n = len(charges)

        confidence_level = 0.95
        z_score = norm.ppf((1 + confidence_level) / 2)  
        margin_of_error = z_score * (std_charge / np.sqrt(n))
        
        ci_lower = mean_charge - margin_of_error
        ci_upper = mean_charge + margin_of_error
        
        print(f"\n📊 Confidence Interval Analysis:")
        print(f"{'─'*60}")
        print(f"Sample size: {n}")
        print(f"Mean charge: ${mean_charge:.2f}")
        print(f"Standard deviation: ${std_charge:.2f}")
        print(f"\nConfidence level: {confidence_level * 100}%")
        print(f"Z-score: {z_score:.4f}")
        print(f"Margin of error: ±${margin_of_error:.2f}")
        print(f"\n95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
        print(f"\n✓ INTERPRETATION:")
        print(f"  We are 95% confident that the true average customer charge")
        print(f"  is between ${ci_lower:.2f} and ${ci_upper:.2f}")
        
        print(f"\n📊 Other Confidence Intervals:")
        print(f"{'─'*60}")
        
        for conf_level in [0.90, 0.95, 0.99]:
            z = norm.ppf((1 + conf_level) / 2)
            moe = z * (std_charge / np.sqrt(n))
            lower = mean_charge - moe
            upper = mean_charge + moe
            print(f"{conf_level*100}% CI: [${lower:.2f}, ${upper:.2f}] (±${moe:.2f})")
 
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(charges, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(mean_charge, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_charge:.2f}')
        plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label=f'Lower CI: ${ci_lower:.2f}')
        plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2, label=f'Upper CI: ${ci_upper:.2f}')
        plt.xlabel('Total Charges ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Customer Charges')
        plt.legend()

        plt.subplot(1, 2, 2)
        conf_levels = [90, 95, 99]
        ci_ranges = []
        
        for conf in conf_levels:
            z = norm.ppf((1 + conf/100) / 2)
            moe = z * (std_charge / np.sqrt(n))
            ci_ranges.append((mean_charge - moe, mean_charge + moe))
        
        y_pos = np.arange(len(conf_levels))
        for i, (lower, upper) in enumerate(ci_ranges):
            plt.plot([lower, upper], [i, i], 'o-', linewidth=3, markersize=8)
            plt.text(mean_charge, i, f' ${mean_charge:.2f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.yticks(y_pos, [f'{c}%' for c in conf_levels])
        plt.xlabel('Charge Amount ($)')
        plt.ylabel('Confidence Level')
        plt.title('Confidence Intervals Comparison')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/confidence_intervals.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: visualizations/confidence_intervals.png")
        plt.close()
    
    def risk_analysis(self):
        """Perform risk analysis using probability distributions"""
        print("\n" + "="*80)
        print("5. RISK ANALYSIS USING PROBABILITY DISTRIBUTIONS")
        print("="*80)
        
        print("\nBusiness Question: What is the probability of high customer service calls (risk indicator)?")

        service_calls = self.df['Customer service calls']
        
        mean_calls = service_calls.mean()
        std_calls = service_calls.std()
        
        print(f"\n📊 Service Calls Distribution:")
        print(f"{'─'*60}")
        print(f"Mean: {mean_calls:.2f} calls")
        print(f"Standard deviation: {std_calls:.2f}")
        print(f"Median: {service_calls.median():.2f}")
        print(f"Maximum: {service_calls.max()} calls")

        prob_4_or_more = (service_calls >= 4).sum() / len(service_calls) * 100
        prob_6_or_more = (service_calls >= 6).sum() / len(service_calls) * 100
        
        print(f"\n📊 Risk Probabilities:")
        print(f"{'─'*60}")
        print(f"P(4+ calls) = {prob_4_or_more:.2f}% → MEDIUM RISK")
        print(f"P(6+ calls) = {prob_6_or_more:.2f}% → HIGH RISK")

        print(f"\n📊 Churn Probability by Service Calls:")
        print(f"{'─'*60}")
        churn_by_calls = self.df.groupby('Customer service calls')['Churn'].apply(
            lambda x: (x == True).sum() / len(x) * 100
        )
        
        for calls, churn_rate in churn_by_calls.items():
            risk_level = "LOW" if calls < 4 else ("MEDIUM" if calls < 6 else "HIGH")
            print(f"  {calls} calls: {churn_rate:.2f}% churn rate [{risk_level} RISK]")

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 3, 1)
        plt.hist(service_calls, bins=range(int(service_calls.max())+2), 
                alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(mean_calls, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_calls:.2f}')
        plt.xlabel('Number of Service Calls')
        plt.ylabel('Frequency')
        plt.title('Distribution of Service Calls')
        plt.legend()

        plt.subplot(1, 3, 2)
        churn_by_calls.plot(kind='bar', color='coral')
        plt.xlabel('Customer Service Calls')
        plt.ylabel('Churn Rate (%)')
        plt.title('Churn Probability by Service Calls')
        plt.xticks(rotation=0)
 
        plt.subplot(1, 3, 3)
        risk_zones = ['Low Risk\n(0-3 calls)', 'Medium Risk\n(4-5 calls)', 'High Risk\n(6+ calls)']
        risk_counts = [
            (service_calls < 4).sum(),
            ((service_calls >= 4) & (service_calls < 6)).sum(),
            (service_calls >= 6).sum()
        ]
        colors = ['green', 'yellow', 'red']
        plt.pie(risk_counts, labels=risk_zones, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Customer Risk Distribution')
        
        plt.tight_layout()
        plt.savefig('visualizations/risk_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: visualizations/risk_analysis.png")
        plt.close()
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS - SUMMARY REPORT")
        print("="*80)
        
        print("\n✓ COMPLETED ANALYSES:")
        print("  1. T-Test: Compared charges between churned vs non-churned customers")
        print("  2. Chi-Square Test: Analyzed relationship between International Plan and Churn")
        print("  3. A/B Testing: Evaluated marketing campaign effectiveness")
        print("  4. Confidence Intervals: Estimated average customer charges with certainty")
        print("  5. Risk Analysis: Assessed customer service call risks")
        
        print("\n✓ KEY BUSINESS INSIGHTS:")

        churn_rate = (self.df['Churn'] == True).sum() / len(self.df) * 100
        print(f"  • Overall churn rate: {churn_rate:.2f}%")

        avg_charge = (self.df['Total day charge'] + self.df['Total eve charge'] + 
                     self.df['Total night charge'] + self.df['Total intl charge']).mean()
        print(f"  • Average customer charge: ${avg_charge:.2f}")

        high_risk = (self.df['Customer service calls'] >= 4).sum()
        print(f"  • High-risk customers (4+ service calls): {high_risk} ({high_risk/len(self.df)*100:.1f}%)")
        
        print("\n✓ VISUALIZATIONS CREATED:")
        print("  1. ttest_visualization.png")
        print("  2. chisquare_visualization.png")
        print("  3. ab_testing_visualization.png")
        print("  4. confidence_intervals.png")
        print("  5. risk_analysis.png")

def main():
    """Main execution function"""
    print("="*80)
    print("CODVEDA INTERNSHIP - TASK 3: STATISTICAL ANALYSIS")
    print("="*80)
    print(f"Execution Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data_path = r"C:\Users\methi\OneDrive\Desktop\Codveda_Level2_Tasks\datasets\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
    
    analysis = StatisticalAnalysis(data_path)

    analysis.data_overview()
    analysis.hypothesis_testing_ttest()
    analysis.hypothesis_testing_chi_square()
    analysis.ab_testing()
    analysis.confidence_intervals()
    analysis.risk_analysis()
    analysis.generate_summary_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review all visualizations created")
    print("2. Document key findings in your report")
    print("3. Prepare LinkedIn post with insights")
    print("4. Share with hashtags: #CodvedaJourney #CodvedaAchievements")

if __name__ == "__main__":
    main()