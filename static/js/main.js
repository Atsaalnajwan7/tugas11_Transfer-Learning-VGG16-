// Auto-dismiss alerts after 4s
document.addEventListener('DOMContentLoaded', () => {
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(a => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(a);
      bsAlert.close();
    }, 4000);
  });
});
