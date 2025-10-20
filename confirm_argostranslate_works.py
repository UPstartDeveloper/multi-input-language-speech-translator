import argostranslate.package, argostranslate.translate

# Download and install a translation model (example: Arabic → English)
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Find the one we want
from_code, to_code = "ar", "en"
package_to_install = next(p for p in available_packages if p.from_code == from_code and p.to_code == to_code)
argostranslate.package.install_from_path(package_to_install.download())

# Translate text
translated = argostranslate.translate.translate("السلام عليكم", "ar", "en")
print(translated)
