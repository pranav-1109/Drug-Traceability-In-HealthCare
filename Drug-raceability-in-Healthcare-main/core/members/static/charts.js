// Access chart data from Django template variables
var company_chart_div = "{{ company_chart|safe }}";
var medicine_chart_div = "{{ medicine_chart|safe }}";

// Parse JSON data
var company_chart_data = JSON.parse(company_chart_div);
var medicine_chart_data = JSON.parse(medicine_chart_div);

function createCompanyChart() {
  var company_chart_element = document.getElementById('company-chart');
  Plotly.newPlot(company_chart_element, company_chart_data);
}

function createMedicineChart() {
  var medicine_chart_element = document.getElementById('medicine-chart');
  Plotly.newPlot(medicine_chart_element, medicine_chart_data);
}

// Call chart creation functions
createCompanyChart();
createMedicineChart();
